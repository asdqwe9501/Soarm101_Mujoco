"""gripperframe 사이트 위치를 목표로 하는 감쇠 최소제곱 IK. 그리퍼는 아래를 향하도록 약하게 유도한다."""

import mujoco
import numpy as np

DOWN = np.array([0.0, 0.0, -1.0])


class IK:
    def __init__(self, model, n_seeds=8, iters=300, damping=1e-5, max_dq=0.1, seed=0):
        self.model = model
        self.data = mujoco.MjData(model)
        self.site = model.site("gripperframe").id
        self.lo = model.jnt_range[:6, 0]
        self.hi = model.jnt_range[:6, 1]
        self.n_seeds, self.iters, self.damping, self.max_dq = n_seeds, iters, damping, max_dq
        self.rng = np.random.default_rng(seed)
        self._jp = np.zeros((3, model.nv))
        self._jr = np.zeros((3, model.nv))

    def _fk(self, q):
        self.data.qpos[:6] = q
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_comPos(self.model, self.data)
        return self.data.site_xpos[self.site], self.data.site_xmat[self.site].reshape(3, 3)

    def _descend(self, target, q, w):
        for _ in range(self.iters):
            p, R = self._fk(q)
            ep = target - p
            eo = np.cross(R[:, 0], DOWN)
            mujoco.mj_jacSite(self.model, self.data, self._jp, self._jr, self.site)
            J = np.vstack([self._jp, w * self._jr])[:, :4]
            e = np.concatenate([ep, w * eo])
            dq = J.T @ np.linalg.solve(J @ J.T + self.damping * np.eye(6), e)
            q[:4] = np.clip(q[:4] + np.clip(dq, -self.max_dq, self.max_dq), self.lo[:4], self.hi[:4])
        p, R = self._fk(q)
        tilt = np.degrees(np.arccos(np.clip(R[:, 0] @ DOWN, -1.0, 1.0)))
        return q, float(np.linalg.norm(target - p)), float(tilt)

    def solve(self, target_xyz, q_init, ori_weight=0.1):
        target = np.asarray(target_xyz, dtype=float)
        starts = [np.array(q_init, dtype=float)]
        for _ in range(self.n_seeds):
            q = np.array(q_init, dtype=float)
            q[0] = np.arctan2(-target[1], target[0])  # pan +방향 = -y
            q[1:4] = self.rng.uniform(self.lo[1:4], self.hi[1:4])
            starts.append(q)
        results = [self._descend(target, s.copy(), ori_weight) for s in starts]
        ok = [r for r in results if r[1] < 0.003]
        return min(ok, key=lambda r: r[2]) if ok else min(results, key=lambda r: r[1])
