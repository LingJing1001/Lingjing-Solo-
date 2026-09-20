from lingjing_solo.evidence import run_ar25_level


def test_ar25_candidate_replay_report(tmp_path):
    result = run_ar25_level(level=1, output_dir=tmp_path, t_limit=5)
    report = result["report_data"]
    assert report["verdict"] == "PASS"
    assert report["criteria"] == {"candidate_present": True, "offline_replay": True, "won": True}
    assert report["metrics"]["replayed_transitions"] == report["metrics"]["candidate_steps"]
    assert (tmp_path / "ar25-l1-offline.jsonl").is_file()
    assert (tmp_path / "ar25-l1-offline.report.json").is_file()
