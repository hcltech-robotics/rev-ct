import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from rev_ct.export import export_artifact
from rev_ct.head import RevHead


def test_export_matches_independent_sklearn_oracle(tmp_path):
    rng = np.random.default_rng(3)
    x = rng.normal(size=(90, 768)).astype(np.float32)
    scaler = StandardScaler().fit(x)
    z = scaler.transform(x)
    clf = LogisticRegression(class_weight="balanced").fit(z, (x[:, 0] > 0))
    iso = IsotonicRegression(out_of_bounds="clip").fit(
        np.array([0.1, 0.3, 0.6, 0.9], dtype=np.float32), [0, 0, 1, 1]
    )
    labels = [f"label {j}" for j in range(18)]
    artifact = {
        "scaler": scaler,
        "clfs": [("lr", clf)] * 18,
        "biases": np.linspace(-1, 1, 18).astype(np.float32),
        "isotonic": [iso] * 18,
        "label_cols": labels,
    }
    export_artifact(artifact, tmp_path, source_revision="test-revision")
    head = RevHead.from_pretrained(tmp_path)
    out = head.predict(x)
    p = clf.predict_proba(scaler.transform(x))[:, 1]
    np.testing.assert_allclose(out.raw_probabilities, np.repeat(p[:, None], 18, axis=1), atol=1e-12)
    np.testing.assert_allclose(out.scores[:, 0], iso.transform(p), atol=1e-7)
    np.testing.assert_array_equal(
        out.decisions, clf.decision_function(z)[:, None] >= artifact["biases"]
    )
