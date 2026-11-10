"""Rule-based specialization choice: prompt, RAG, LoRA, or full fine-tuning."""
def recommend(knowledge_changes_often, private_facts, behavior_gap, broad_capability_gap, data_examples):
    if knowledge_changes_often or private_facts: return "RAG"
    if not behavior_gap and not broad_capability_gap: return "prompting"
    if behavior_gap and data_examples>=100: return "LoRA"
    if broad_capability_gap and data_examples>=10000: return "full_fine_tuning"
    return "collect_evaluation_and_training_data"
def matrix():
    return {"prompting":("instructions","no weight update","limited durable behavior"),
            "RAG":("changing/private facts","retrieval pipeline","retrieval and citation failures"),
            "LoRA":("bounded behavior/style","adapter weights","data coverage and forgetting"),
            "full_fine_tuning":("broad capability change","large labeled corpus","high cost and regression risk")}
if __name__ == "__main__": print(matrix()); print(recommend(True,True,False,False,0))
