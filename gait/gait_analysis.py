import numpy as np

class GaitAnalyzer:
    def __init__(self, foot_z_indices=(4,5), frame_rate=30):
        self.left_idx, self.right_idx = foot_z_indices
        self.frame_rate = frame_rate

    def detect_events(self, foot_z_seq, vz_thresh=0.02, z_thresh=0.05):
        N = foot_z_seq.shape[0]
        vz = np.zeros_like(foot_z_seq)
        vz[1:] = foot_z_seq[1:] - foot_z_seq[:-1]
        vz /= (1/self.frame_rate)
        left_contact  = (foot_z_seq[:,0] < z_thresh) & (vz[:,0] <= vz_thresh)
        right_contact = (foot_z_seq[:,1] < z_thresh) & (vz[:,1] <= vz_thresh)
        left_off  = vz[:,0] > vz_thresh
        right_off = vz[:,1] > vz_thresh
        return left_contact, left_off, right_contact, right_off

    def split_gait_cycles(self, contact_array):
        indices = np.where(contact_array)[0]
        if len(indices) < 2: return []
        cycles = []
        for i in range(len(indices)-1):
            cycles.append((indices[i], indices[i+1]))
        return cycles

    def compute_step_length(self, foot_pos_seq, start_idx, end_idx):
        start = foot_pos_seq[start_idx]
        end   = foot_pos_seq[end_idx]
        return np.linalg.norm(end[[0,1]] - start[[0,1]])

    def analyze(self, skeleton_seq):
        foot_z = skeleton_seq[:, [self.left_idx, self.right_idx], 2]
        left_contact, left_off, right_contact, right_off = self.detect_events(foot_z)
        left_cycles  = self.split_gait_cycles(left_contact)
        right_cycles = self.split_gait_cycles(right_contact)
        gait_cycles = []
        for (s,e) in left_cycles:
            duration = (e-s)/self.frame_rate
            step_len = self.compute_step_length(skeleton_seq[:, self.left_idx, :], s, e)
            gait_cycles.append({'foot':'left','start_frame':s,'end_frame':e,'duration':duration,'step_length':step_len})
        for (s,e) in right_cycles:
            duration = (e-s)/self.frame_rate
            step_len = self.compute_step_length(skeleton_seq[:, self.right_idx, :], s, e)
            gait_cycles.append({'foot':'right','start_frame':s,'end_frame':e,'duration':duration,'step_length':step_len})
        return gait_cycles

if __name__ == "__main__":
    skeleton_seq = np.random.rand(50,8,3)
    analyzer = GaitAnalyzer(foot_z_indices=(4,5), frame_rate=30)
    cycles = analyzer.analyze(skeleton_seq)
    for c in cycles:
        print(f"{c['foot']} foot: frames {c['start_frame']}–{c['end_frame']}, "
              f"duration {c['duration']:.2f}s, step length {c['step_length']:.3f}m")