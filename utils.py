import math
import os

import matplotlib.pyplot as plt
import torch
import cv2 as cv
import numpy as np
from PIL import Image


def show_np(img, title, save_path=None):
    plt.figure()
    plt.title(title)
    plt.imshow(img)
    if save_path:
        plt.savefig(save_path)
    plt.show()
    plt.close()


def show_torch(img, title, save_path=None):
    if len(img.shape) == 4:
        img = img[0]
    if img.shape[0] == 1:
        img = img[0]
    show_np(img.numpy(), title, save_path)


def avg_loss(data):
    return (torch.linalg.norm(data, dim=1) ** 2).sum() / data.shape[0]


def split_points(tentative_matches, kps0, kps1):
    src_pts = np.float32([kps0[m.queryIdx].pt for m in tentative_matches]).reshape(-1, 2)
    dst_pts = np.float32([kps1[m.trainIdx].pt for m in tentative_matches]).reshape(-1, 2)
    kps0 = [kps0[m.queryIdx] for m in tentative_matches]
    kps1 = [kps1[m.trainIdx] for m in tentative_matches]
    return src_pts, dst_pts, kps0, kps1


def get_tentatives(kpts0, desc0, kpts1, desc1, ratio_threshold, space_dist_th=None):
    matcher = cv.BFMatcher(crossCheck=False)
    knn_matches = matcher.knnMatch(desc0, desc1, k=2)
    matches2 = matcher.match(desc1, desc0)

    tentative_matches = []
    for m, n in knn_matches:
        if matches2[m.trainIdx].trainIdx != m.queryIdx:
            continue

        if space_dist_th:
            x = kpts0[m.queryIdx].pt
            y = kpts1[m.trainIdx].pt
            dist = math.sqrt((x[0] - y[0]) ** 2 + (x[1] - y[1]) ** 2)
            if dist > space_dist_th:
                continue

        if m.distance < ratio_threshold * n.distance:
            tentative_matches.append(m)

    src, dst, kpts0, kpts1 = split_points(tentative_matches, kpts0, kpts1)
    return src, dst, kpts0, kpts1, tentative_matches


def csv2latex(str):

    ret = ""
    for line in str.split("\n"):
        tokens = [l.strip() for l in line.split("\t") if len(l.strip()) > 0]
        ret += " & ".join(tokens) + " \\\\ \n\\hline\n"
    return ret

def csv2latex_exps():

    r = csv2latex("""OpenCV		VLFeat		Kornia		
    scale	baseline	correction 0.25	baseline	correction	baseline	correction	bilinear
    0.1	0.885	0.709	36.345	38.184	N/A	N/A	N/A
    0.2	0.402	0.249	1.070	0.840	0.513	0.611	0.478
    0.3	0.312	0.255	0.496	0.311	0.303	0.334	0.371
    0.4	0.130	0.235	0.243	0.130	0.309	0.289	0.202
    0.5	0.172	0.006	0.343	0.167	0.209	0.196	0.065
    0.6	0.372	0.240	0.497	0.359	0.297	0.230	0.275
    0.7	0.280	0.233	0.349	0.268	0.321	0.221	0.292
    0.8	0.233	0.244	0.247	0.234	0.318	0.282	0.243
    0.9	0.217	0.248	0.171	0.196	0.322	0.290	0.169
    SUM	3.005	2.419	39.761	40.689	2.593	2.454	2.095
    SUM (w/o 0.1)	2.120	1.710	3.416	2.505	2.593	2.454	2.095""")
    print(r)

    r = csv2latex("""			OpenCV			VLFeat			Kornia					
    			baseline			baseline			baseline:  nearest	bilinear	 nearest	bilinear	 nearest	bilinear
    		correction[px.]	0.000	0.250	-0.250	0.000	0.250	-0.250	0.000	0.000	0.250	0.250	-0.250	-0.250
    scale	rotation	img 1 vs img												
    0.822	31.494	2.000	1.332	1.436	1.246	1.093	1.145	1.062	1.258	1.369	1.433	1.224	1.065	1.090
    0.541	-150.235	3.000	2.178	2.633	1.762	1.652	2.006	1.392	2.212	1.990	2.422	2.527	2.272	2.243
    0.404	119.795	4.000	1.340	1.729	0.993	0.963	1.309	0.740	1.350	1.143	1.575	1.502	1.148	0.920
    0.332	22.770	5.000	0.712	0.809	0.712	0.699	0.701	0.825	0.711	0.723	0.851	0.904	0.802	0.646
    0.244	-153.391	6.000	1.257	1.596	0.981	0.996	1.267	0.884	1.352	1.044	1.648	1.305	1.158	1.251
    		SUM	6.819	8.203	5.695	5.403	6.426	4.903	6.883	6.268	7.929	7.461	6.447	6.150""")
    print(r)

    r = csv2latex("""			OpenCV			VLFeat		
    			baseline			baseline		
    		correction[px.]	0.000	0.250	-0.250	0.000	0.250	-0.250
    scale	rotation	img 1 vs img						
    0.885	13.972	2.000	0.313	0.348	0.299	0.395	0.298	0.373
    0.736	39.582	3.000	0.286	0.469	0.227	0.211	0.406	0.474
    0.532	79.841	4.000	1.327	1.388	1.331	0.621	0.477	1.575
    0.424	-8.361	5.000	0.459	1.116	1.122	1.931	1.187	2.164
    0.358	40.536	6.000	5.696	5.617	5.800	5.743	5.588	6.202
    		SUM	8.082	8.938	8.778	8.901	7.956	10.788""")
    print(r)

    r = csv2latex("""			Kornia					
    			baseline:  nearest	bilinear	 nearest	bilinear	 nearest	bilinear
    		correction[px.]	0.000	0.000	0.250	0.250	-0.250	-0.250
    scale	rotation	img 1 vs img						
    0.885	13.972	2.000	0.626	0.505	0.519	0.294	0.436	0.489
    0.736	39.582	3.000	0.313	0.324	0.432	0.568	0.267	0.289
    0.532	79.841	4.000	1.635	2.043	1.482	0.975	1.440	1.412
    0.424	-8.361	5.000	0.869	2.105	1.350	2.069	0.626	1.254
    0.358	40.536	6.000	6.004	6.747	6.075	4.490	5.079	5.562
    		SUM	9.447	11.724	9.857	8.396	7.848	9.007""")
    print(r)

    r = csv2latex("""	OpenCV			VLFeat		
    	baseline					
    correction[px.]	0.000	0.250	-0.250	0.000	0.250	-0.250
    degrees						
    90.000	0.500	1.000	0.002	0.002	0.499	0.501
    180.000	0.705	1.412	0.003	0.005	0.707	0.707
    270.000	0.500	1.000	0.001	0.004	0.502	0.499
    SUM	1.705	3.412	0.006	0.010	1.708	1.707""")
    print(r)

    r = csv2latex("""	Kornia					
    	baseline:  nearest	bilinear	 nearest	bilinear	 nearest	bilinear
    correction[px.]	0.000	0.000	0.250	0.250	-0.250	-0.250
    degrees						
    90.000	0.516	0.471	0.999	0.945	0.082	0.098
    180.000	0.675	0.729	1.384	1.413	0.121	0.093
    270.000	0.455	0.437	1.022	0.993	0.068	0.052
    SUM	1.646	1.637	3.405	3.351	0.271	0.243""")
    print(r)

    r = csv2latex("""	OpenCV			VLFeat		
    	baseline					
    correction[px.]	0.000	0.250	-0.250	0.000	0.250	-0.250
    scale						
    0.100	0.885	0.709	1.134	36.345	38.184	37.093
    0.200	0.402	0.249	0.642	1.070	0.840	1.323
    0.300	0.312	0.255	0.496	0.496	0.311	0.720
    0.400	0.130	0.235	0.247	0.243	0.130	0.440
    0.500	0.172	0.006	0.349	0.343	0.167	0.520
    0.600	0.372	0.240	0.510	0.497	0.359	0.636
    0.700	0.280	0.233	0.357	0.349	0.268	0.441
    0.800	0.233	0.244	0.244	0.247	0.234	0.276
    0.900	0.217	0.248	0.187	0.171	0.196	0.151
    SUM	3.005	2.419	4.166	39.761	40.689	41.601
    SUM (w/o 0.1)	2.120	1.710	3.032	3.416	2.505	4.508""")
    print(r)

    r = csv2latex("""	Kornia					
    	baseline:  nearest	bilinear	 nearest	bilinear	 nearest	bilinear
    correction[px.]	0.000	0.000	0.250	0.250	-0.250	-0.250
    scale						
    0.100	N/A	N/A	N/A	N/A	N/A	N/A
    0.200	0.513	0.478	0.611	0.568	0.746	0.537
    0.300	0.303	0.371	0.334	0.468	0.578	0.369
    0.400	0.309	0.202	0.289	0.367	0.356	0.133
    0.500	0.209	0.065	0.196	0.153	0.249	0.185
    0.600	0.297	0.275	0.230	0.279	0.611	0.442
    0.700	0.321	0.292	0.221	0.217	0.319	0.325
    0.800	0.318	0.243	0.282	0.254	0.314	0.302
    0.900	0.322	0.169	0.290	0.195	0.263	0.290
    SUM	2.593	2.095	2.454	2.500	3.436	2.583
    SUM (w/o 0.1)	2.593	2.095	2.454	2.500	3.436	2.583""")
    print(r)

    r = csv2latex("""0.500	0.002	0.002	0.501	0.516	0.082	0.001	0.500
0.705	0.003	0.005	0.707	0.675	0.121	0.002	0.707
0.500	0.001	0.004	0.499	0.455	0.068	0.002	0.499
1.705	0.006	0.010	1.707	1.646	0.271	0.005	1.706""")
    print(r)


if __name__ == "__main__":

    csv2latex_exps()

    # print("start")
    # dir = "demo_imgs/lowe_all/imgs"
    # f_out = "demo_imgs/lowe_all/run.bat"
    # with open(f_out, "w") as f:
    #     files = [f for f in sorted(list(os.listdir(dir))) if f.endswith("pgm")]
    #     for file in files:
    #         f.write(f"siftWin32 < imgs\\{file} > keys\\{file}.key\n")
    # print("end")