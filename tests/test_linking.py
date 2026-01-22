import numpy as np
from toy3body_topo.linking import safe_linking_integer


def make_circle_xy(n=512, R=1.0, z=0.0, center=(0.0,0.0,0.0)):
    t = np.linspace(0.0, 2*np.pi, n, endpoint=False)
    x = center[0] + R*np.cos(t)
    y = center[1] + R*np.sin(t)
    z = np.full_like(x, z) + center[2]
    C = np.vstack([x,y,z]).T
    C = np.vstack([C, C[0]])
    return C.astype(float)


def make_circle_xz(n=512, R=1.0, y=0.0, center=(0.5,0.0,0.0)):
    t = np.linspace(0.0, 2*np.pi, n, endpoint=False)
    x = center[0] + R*np.cos(t)
    y = np.full_like(x, y) + center[1]
    z = center[2] + R*np.sin(t)
    C = np.vstack([x,y,z]).T
    C = np.vstack([C, C[0]])
    return C.astype(float)


def test_hopf_link():
    C1 = make_circle_xy()
    C2 = make_circle_xz()
    ok, lk_int, lk_raw = safe_linking_integer(C1, C2, round_tol=0.2, min_sep=1e-3)
    assert ok, f"linking not stable: raw={lk_raw}"
    assert abs(lk_int) == 1, f"expected |Lk|=1 got {lk_int}, raw={lk_raw}"


def test_unlinked():
    C1 = make_circle_xy(center=(0.0,0.0,0.0))
    C2 = make_circle_xy(center=(0.0,0.0,5.0))
    ok, lk_int, lk_raw = safe_linking_integer(C1, C2, round_tol=0.2, min_sep=1e-3)
    assert ok
    assert lk_int == 0, f"expected 0 got {lk_int}, raw={lk_raw}"


if __name__ == "__main__":
    test_hopf_link()
    test_unlinked()
    print("OK")
