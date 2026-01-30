# xlite 投机推理 (Speculative Decoding) 设计方案

## 1. 背景

投机推理通过"小模型快速起草 + 大模型并行验证"来加速自回归生成，在不改变输出分布的前提下，减少大模型前向传播次数。

**预期收益**：1.5x - 3x 推理加速（取决于 draft model 准确率和 k 值）

---

## 2. 核心设计

### 2.1 架构概览

```
┌─────────────────────────────────────────────────────────────┐
│                    SpeculativeDecoder                        │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐    draft tokens    ┌─────────────────┐    │
│  │ DraftModel  │ ──────────────────>│  TargetModel    │    │
│  │ (小模型)     │                    │  (大模型)        │    │
│  │ 快速生成 k   │<──────────────────│  并行验证        │    │
│  │ 个候选token │   accept/reject    │                 │    │
│  └─────────────┘                    └─────────────────┘    │
│         │                                   │               │
│         └───────────── KV Cache Manager ────┘               │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 核心组件

```cpp
// speculative_decoder.h

class SpeculativeDecoder {
public:
    struct Config {
        int num_draft_tokens = 5;      // k: 每次起草的 token 数
        float temperature = 1.0f;
        int max_seq_len = 4096;
        bool use_tree_attention = false;  // 是否使用树形注意力
    };

    SpeculativeDecoder(
        std::shared_ptr<Model> draft_model,
        std::shared_ptr<Model> target_model,
        Config config
    );

    // 主要接口
    std::vector<int> generate(
        const std::vector<int>& prompt_tokens,
        int max_new_tokens
    );

private:
    // 起草阶段：小模型生成 k 个候选
    std::vector<int> draft(int k);
    
    // 验证阶段：大模型并行验证
    VerifyResult verify(const std::vector<int>& draft_tokens);
    
    // 采样与拒绝
    int sample_with_rejection(
        const Tensor& target_probs,
        const Tensor& draft_probs,
        int draft_token
    );
};
```

### 2.3 验证算法 (核心)

```cpp
struct VerifyResult {
    int num_accepted;           // 接受的 token 数量
    std::vector<int> accepted;  // 接受的 tokens
    int bonus_token;            // 额外采样的 token (来自拒绝位置)
};

VerifyResult verify(const std::vector<int>& draft_tokens) {
    // 1. 大模型一次 forward，计算所有位置的 logits
    //    输入: [prefix..., draft_0, draft_1, ..., draft_{k-1}]
    //    输出: logits for positions [len(prefix), ..., len(prefix)+k]
    
    Tensor target_logits = target_model_->forward(
        concat(prefix_tokens_, draft_tokens)
    );
    
    VerifyResult result;
    
    // 2. 逐个验证
    for (int i = 0; i < draft_tokens.size(); i++) {
        float p_target = softmax(target_logits[i])[draft_tokens[i]];
        float p_draft = draft_probs_[i][draft_tokens[i]];
        
        float accept_prob = std::min(1.0f, p_target / p_draft);
        
        if (random() < accept_prob) {
            // 接受
            result.accepted.push_back(draft_tokens[i]);
        } else {
            // 拒绝：从修正分布中采样
            Tensor corrected = max(0, target_logits[i] - draft_probs_[i]);
            result.bonus_token = sample(normalize(corrected));
            break;
        }
    }
    
    // 3. 如果全部接受，额外采样一个 bonus token
    if (result.accepted.size() == draft_tokens.size()) {
        result.bonus_token = sample(softmax(target_logits.back()));
    }
    
    result.num_accepted = result.accepted.size();
    return result;
}
```

---

## 3. 昇腾 NPU 适配要点

### 3.1 KV Cache 管理

```cpp
class SpeculativeKVCache {
public:
    // Draft model 和 Target model 共享 prefix 部分的 KV Cache
    // Draft 阶段产生的 KV 需要在验证后选择性保留
    
    void fork_for_draft();           // 为 draft 创建临时分支
    void commit(int num_accepted);   // 提交接受的 tokens
    void rollback();                 // 回滚拒绝的 tokens
    
private:
    // 昇腾上使用 aclnn 管理显存
    aclTensor* prefix_kv_;
    aclTensor* draft_kv_;  // 临时缓冲
};
```

### 3.2 NPU 算子调度

```cpp
// 昇腾上的关键优化：验证阶段的 batch attention

// 传统做法：k 次 attention forward
// 优化做法：1 次 batched attention (验证所有 draft tokens)

// 使用 aclnnFlashAttention 或自定义融合算子
aclnnStatus spec_verify_attention(
    const aclTensor* query,      // [batch=1, seq=k, head, dim]
    const aclTensor* key,        // [batch=1, seq=prefix+k, head, dim]  
    const aclTensor* value,
    const aclTensor* attn_mask,  // causal mask for speculative
    aclTensor* output
);
```

### 3.3 流水线优化

```
Timeline (异步执行):
────────────────────────────────────────────────────────>
Draft:    [gen_0][gen_1][gen_2][gen_3][gen_4]
                                              \
Target:                                        [verify all]
                                                    \
Next:                                                [draft...]

通过 ACL Stream 实现 draft 和 verify 的部分重叠
```

---

## 4. 文件结构建议

```
xlite/
├── include/
│   └── xlite/
│       ├── speculative/
│       │   ├── speculative_decoder.h
│       │   ├── draft_model.h
│       │   ├── kv_cache_manager.h
│       │   └── sampler.h
│       └── ...
├── src/
│   └── speculative/
│       ├── speculative_decoder.cpp
│       ├── draft_model.cpp
│       ├── kv_cache_manager.cpp
│       ├── verify_kernel.cpp        # NPU verify 算子
│       └── sampler.cpp
├── tests/
│   └── speculative/
│       ├── test_speculative_decoder.cpp
│       └── test_kv_cache.cpp
└── examples/
    └── speculative_llama.cpp
```

---

## 5. 配置示例

```yaml
# config/speculative_config.yaml
speculative:
  enabled: true
  draft_model: "llama-68m"           # 小模型路径
  target_model: "llama-7b"           # 大模型路径
  num_draft_tokens: 5                # k 值
  
  # 高级选项
  tree_attention: false              # 树形投机 (可选)
  dynamic_k: true                    # 动态调整 k 值
  
  # NPU 特定
  draft_device: "npu:0"
  target_device: "npu:0"             # 可分卡部署
  enable_async: true                 # 异步流水线
```

---

## 6. 实现路线图

### Phase 1: 基础功能 (2-3 周)
- [ ] SpeculativeDecoder 框架
- [ ] 基本的 draft/verify 循环
- [ ] KV Cache fork/commit/rollback
- [ ] 单元测试

### Phase 2: NPU 优化 (2 周)
- [ ] Batched verify attention 算子
- [ ] KV Cache 显存优化
- [ ] ACL Stream 流水线

### Phase 3: 高级特性 (可选)
- [ ] Tree attention (更高并行度)
- [ ] 动态 k 值调整
- [ ] Multi-draft (多个 draft model ensemble)

---

## 7. 参考资料

- [Speculative Decoding 原始论文](https://arxiv.org/abs/2211.17192)
- [vLLM Speculative Decoding 实现](https://github.com/vllm-project/vllm)
- [Medusa: 多头 Draft](https://arxiv.org/abs/2401.10774)
- [昇腾 CANN 文档](https://www.hiascend.com/document)

---

## 8. 贡献流程

1. Fork GVirt 仓库
2. 创建 feature 分支: `git checkout -b feature/speculative-decoding`
3. 按 Phase 实现并提交
4. 提交 PR 到 openeuler/GVirt
5. 参与社区 Code Review

---

*设计者: [Your Name]*  
*日期: 2026-01-30*
