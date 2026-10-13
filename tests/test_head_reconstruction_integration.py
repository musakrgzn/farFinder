import os
import json

import pytest

from farfinder.head.pipeline import reconstruct
from farfinder.head.synthetic import generate_session,validate_reconstruction


@pytest.mark.skipif(os.environ.get("FARFINDER_RUN_RECONSTRUCTION_TEST")!="1",
                    reason="Opt-in real COLMAP integration test; run the documented demo or set FARFINDER_RUN_RECONSTRUCTION_TEST=1")
@pytest.mark.parametrize("capture_mode",["moving-camera","head-rotation"])
def test_real_colmap_synthetic_reconstruction(tmp_path,capture_mode):
    session=generate_session(tmp_path,orbit_views=24,capture_mode=capture_mode)
    report=reconstruct(session)
    validation=validate_reconstruction(session)
    assert report["reconstruction_status"]=="SUCCEEDED"
    assert validation["registered_views"]>=len(session.metadata["frames"])*0.65
    assert not report["metric_scale"]["valid"]
    assert report["capture_mode"]==capture_mode
    if capture_mode=="head-rotation":
        assert report["experimental_capture"]
        assert report["pose_coordinate_frame"]=="head_relative"
        assert "stationary board" in report["metric_scale"]["reason"]
    assert max(validation["extent_relative_errors"])<0.1
    for name in ("pointcloud.ply","head_mesh.ply","camera_poses.json","landmarks.json",
                 "measurements.json","measurements.csv","quality_report.json"):
        assert (session.path/name).is_file()
    results=json.loads((session.path/"measurements.json").read_text())["measurements"]
    assert any(item["distance_value"] is not None for item in results)
    assert all(item["distance_unit"]=="reconstruction_units" for item in results)
