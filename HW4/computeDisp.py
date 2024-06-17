import numpy as np
import cv2.ximgproc as xip
import cv2
def census_cost(compare_pattern_L, compare_pattern_R):
    pattern_hamdist = np.sum(np.abs(compare_pattern_L - compare_pattern_R), axis=1)

    return pattern_hamdist


def computeDisp(Il, Ir, max_disp):
    h, w, ch = Il.shape
    labels = np.zeros((h, w), dtype=np.float32)
    Il = Il.astype(np.float32)
    Ir = Ir.astype(np.float32)


    # parameters
    window_size = 3


    padded_l= cv2.copyMakeBorder(Il,1,1,1,1,cv2.BORDER_CONSTANT,value=[0,0,0])
    padded_r= cv2.copyMakeBorder(Ir,1,1,1,1,cv2.BORDER_CONSTANT,value=[0,0,0])

    
    # >>> Cost Computation
    # TODO: Compute matching cost
    # [Tips] Census cost = Local binary pattern -> Hamming distance
    # [Tips] Set costs of out-of-bound pixels = cost of closest valid pixel
    # [Tips] Compute cost both Il to Ir and Ir to Il for later left-right consistency
    cost_list_L = np.zeros((h, w, max_disp), dtype=np.float32)
    cost_list_R = np.zeros((h, w, max_disp), dtype=np.float32)
    compare_pattern_IL, compare_pattern_IR = [], []
    for i in range(h):
        for j in range(w):
            tmp_l = padded_l[i : i + window_size, j : j + window_size, :].copy()
            tmp_r = padded_r[i : i + window_size, j : j + window_size, :].copy()
            for k in range(ch):
                middle_value = tmp_l[window_size//2, window_size//2, k]
                tmp_l[:, :, k] = np.where(tmp_l[:, :, k] >= middle_value, 0, 1)
                middle_value = tmp_r[window_size//2, window_size//2, k]
                tmp_r[:, :, k] = np.where(tmp_r[:, :, k] >= middle_value, 0, 1)
            compare_pattern_IL.append(tmp_l)
            compare_pattern_IR.append(tmp_r)

   
    compare_pattern_IL = np.array(compare_pattern_IL).reshape(h, w, -1) # (h, w, 27)
    compare_pattern_IR = np.array(compare_pattern_IR).reshape(h, w, -1) # (h, w, 27)

    for i in range(h):
        for j in range(w):
            # left to right
            if j < max_disp - 1: 
                compare_pattern_L = compare_pattern_IL[i, j].copy()[np.newaxis, :]
                compare_pattern_R = np.flip(compare_pattern_IR[i, : j+1].copy(), 0)
                pattern_hamdist = census_cost(compare_pattern_L, compare_pattern_R)

                cost_list_L[i, j, :j+1] = pattern_hamdist
                cost_list_L[i, j, j+1:] = cost_list_L[i, j, j]
            else:
                compare_pattern_L = compare_pattern_IL[i, j].copy()[np.newaxis, :]
                compare_pattern_R = np.flip(compare_pattern_IR[i, (j - max_disp + 1): j + 1].copy(), 0)
                pattern_hamdist = census_cost(compare_pattern_L, compare_pattern_R)
                cost_list_L[i, j, :] = pattern_hamdist
            
            # right to left
            if j + max_disp > w:
                compare_pattern_L = compare_pattern_IL[i, j : w].copy()
                compare_pattern_R = compare_pattern_IR[i, j].copy()[np.newaxis, :]
                pattern_hamdist = census_cost(compare_pattern_L, compare_pattern_R)
                cost_list_R[i, j, :w - j] = pattern_hamdist
                cost_list_R[i, j, w - j:] = cost_list_R[i, j, w - j - 1]
            else:
                compare_pattern_L = compare_pattern_IL[i, j : j + max_disp].copy()
                compare_pattern_R = compare_pattern_IR[i, j].copy()[np.newaxis, :]
                pattern_hamdist = census_cost(compare_pattern_L, compare_pattern_R)
                cost_list_R[i, j, :] = pattern_hamdist
    #print(cost_list_L)
    # >>> Cost Aggregation
    # TODO: Refine the cost according to nearby costs
    # [Tips] Joint bilateral filter (for the cost of each disparity)
    guidance = cv2.cvtColor(Il,cv2.COLOR_BGR2GRAY)
    
    for d in range(max_disp):
        #cost_list_L[:, :, d] = cv2.boxFilter(cost_list_L[:,:,d],-1,(3,3))
        cost_list_L[:, :, d] = xip.jointBilateralFilter(Il, cost_list_L[:, :, d], 20, 10,24)
        #cost_list_L[:,:,d] = cv2.ximgproc.guidedFilter(guidance,cost_list_L[:,:,d],6,0.1,-1)
        #cost_list_R[:, :, d] = cv2.boxFilter(cost_list_R[:,:,d],-1,(3,3))
        cost_list_R[:, :, d] = xip.jointBilateralFilter(Il, cost_list_R[:, :, d], 20, 10,24)
        #cost_list_R[:,:,d] = cv2.ximgproc.guidedFilter(guidance,cost_list_R[:,:,d],6,0.1,-1)
    # >>> Disparity Optimization
    # TODO: Determine disparity based on estimated cost.
    # [Tips] Winner-take-all
    winner_L = np.argmin(cost_list_L, axis=2)
    winner_R = np.argmin(cost_list_R, axis=2)
    # >>> Disparity Refinement
    # TODO: Do whatever to enhance the disparity map
    # [Tips] Left-right consistency check -> Hole filling -> Weighted median filtering
    for i in range(h):
        for j in range(w):
            if j - winner_L[i, j] > 0 and winner_L[i, j] == winner_R[i, j - winner_L[i, j]]:
                continue
            else:
                winner_L[i, j]=-1
    
    for i in range(h):
        for j in range(w):
            if winner_L[i, j] == -1:
                l_idx = j - 1
                r_idx = j + 1
                while l_idx >= 0 :
                    if winner_L[i, l_idx] != -1:
                        break
                    else:
                        l_idx -= 1
                
                if l_idx < 0:
                    FL = max_disp+1
                else:
                    FL = winner_L[i, l_idx]

                while r_idx < w :
                    if winner_L[i, r_idx] != -1:
                        break
                    else:
                        r_idx += 1

                if r_idx > w - 1:
                    FR = max_disp+1
                else:
                    FR = winner_L[i, r_idx]
                winner_L[i, j] = min(FL, FR)

    labels = xip.weightedMedianFilter(Il.astype(np.uint8), winner_L.astype(np.uint8), 15, 1)
    return labels.astype(np.uint8)