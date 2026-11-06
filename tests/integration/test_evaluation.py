from evaluation import run_eval


def test_evaluation_pipeline_never_serves_unsafe_hits():
    rows = run_eval.run_pairs(0.8, live=False)
    sm = run_eval.summarize(rows)
    assert sm["unsafe_hits_guarded"] == 0 and sm["correct_outcomes"] == sm["cases"]
    assert sm["unsafe_hits_similarity_only"] > 0  # the guards matter: similarity alone would have served wrong answers


def test_cost_model_break_even():
    cc = run_eval.cost_comparison(0.5)
    assert cc["net_savings_usd"] > 0 and run_eval.cost_comparison(0.0)["net_savings_usd"] < 0
