"""Acceptance accounting and a transparent speculative-decoding speed bound."""
def account(draft_tokens, accepted_prefix, draft_cost, verify_cost, target_step_cost):
    if not 0 <= accepted_prefix <= draft_tokens: raise ValueError("accepted prefix out of range")
    emitted=accepted_prefix + int(accepted_prefix < draft_tokens)
    baseline=max(1,emitted)*target_step_cost
    speculative=draft_tokens*draft_cost + verify_cost
    return {"accepted":accepted_prefix,"rejected":draft_tokens-accepted_prefix,"emitted_per_cycle":emitted,
            "acceptance_rate":accepted_prefix/max(1,draft_tokens),"baseline_cost":baseline,"speculative_cost":speculative,
            "speedup_bound":baseline/speculative if speculative else float("inf")}
if __name__ == "__main__":
    for a in range(5): print(account(4,a,.15,.8,1.0))
