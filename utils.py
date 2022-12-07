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

    def format_number(s):
        s = s.strip()
        set1 = set(s)
        if len(set1) > 0 and set1.issubset(set(list("0123456789."))):
            s = "{:.3f}".format(float(s))
        return s

    ret = ""
    for line in str.split("\n"):
        tokens = [format_number(l) for l in line.split("\t") if len(l.strip()) > 0]
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


    r = csv2latex("""0.5	0.002	0	0.5	0.001	0.5	0.534	0.193
0.705	0.003	0.001	0.707	0.002	0.707	0.707	0.001
0.5	0.001	0.001	0.499	0.002	0.499	0.537	0.077
1.705	0.006	0.002	1.706	0.005	1.706	1.778	0.271""")
    print(r)

    r = csv2latex("""0.591	0.29	0.922	0.608	0.91	0.598	N/A	N/A	N/A
0.45	0.199	0.723	0.449	0.729	0.455	0.681	0.284	0.23
0.179	0.347	0.229	0.183	0.219	0.178	0.225	0.362	0.274
0.169	0.1	0.362	0.166	0.365	0.168	0.226	0.133	0.138
0.392	0.246	0.56	0.396	0.568	0.403	0.428	0.342	0.256
0.225	0.099	0.36	0.223	0.363	0.226	0.269	0.251	0.193
0.351	0.342	0.392	0.354	0.39	0.353	0.427	0.391	0.349
0.171	0.2	0.169	0.169	0.169	0.168	0.193	0.261	0.168
0.293	0.28	0.31	0.293	0.305	0.288	0.323	0.344	0.317
2.821	2.103	4.027	2.841	4.018	2.837	2.772	2.368	1.925""")
    print(r)


    r = csv2latex("""openCV					
experiment: bark			experiment: bark		
OWN BUILD			OWN BUILD		
NEAREST	NEAREST	NEAREST	LINEAR	LINEAR	LINEAR
OpenCV: 0.0	OpenCV: 0.25	OpenCV: -0.25	OpenCV: 0.0	OpenCV: 0.25	OpenCV: -0.25
MAE			MAE		
1.332	1.436	1.246	1.208	1.27	1.166
2.178	2.633	1.762	1.963	2.391	1.588
1.34	1.729	0.993	1.164	1.553	0.827
0.712	0.809	0.712	0.612	0.771	0.522
1.257	1.596	0.981	1.029	1.333	0.84
6.819	8.203	5.694	5.976	7.318	4.943
					
					
experiment: bark			experiment: bark		
OWN BUILD			OWN BUILD		
LANCZOS	LANCZOS	LANCZOS	CUBIC	CUBIC	CUBIC
OpenCV: 0.0	OpenCV: 0.25	OpenCV: -0.25	OpenCV: 0.0	OpenCV: 0.25	OpenCV: -0.25
MAE			MAE		
0.888	0.972	0.833	0.939	1.005	0.9
1.9	2.33	1.526	2.001	2.428	1.626
1.089	1.474	0.762	1.193	1.579	0.862
0.672	0.829	0.576	0.685	0.841	0.593
1.043	1.341	0.863	1.025	1.33	0.833
5.592	6.946	4.56	5.843	7.183	4.814""")
    print(r)


    r = csv2latex("""bark										
HLOC SIFT: adj=0.0, conf=opencv_like_conf	HLOC SIFT: adj=0.25, conf=opencv_like_conf	HLOC SIFT: adj=-0.25, conf=opencv_like_conf	Lowe SIFT: [0.0, 0.0]	Lowe SIFT: [0.25, 0.25]	Lowe SIFT: [-0.25, -0.25]
MAE			MAE		
1.205	1.288	1.14	1.176	1.219	1.153
1.694	2.107	1.349	1.689	2.096	1.353
0.977	1.325	0.743	0.948	1.293	0.725
0.715	0.725	0.837	0.709	0.694	0.837
0.993	1.258	0.89	1.009	1.279	0.895
5.584	6.703	4.959	5.531	6.581	4.963""")
    print(r)


    r = csv2latex("""experiment: bark					
NEAREST			BILINEAR		
	SIFT kornia nearest [0.25 0.25]		SIFT kornia bilinear [0. 0.]		SIFT kornia bilinear [-0.25 -0.25]
SIFT kornia nearest [0. 0.]		SIFT kornia nearest [-0.25 -0.25]		SIFT kornia bilinear [0.25 0.25]	
MAE					
1.251	1.361	0.91	1.452	0.826	1.269
2.085	2.621	1.712	2.289	2.602	1.378
1.491	1.819	1.148	1.263	1.624	0.932
0.771	0.861	0.84	0.679	0.734	0.713
1.335	1.628	0.979	1.12	1.424	1.055
6.933	8.29	5.589	6.803	7.21	5.347""")
    print(r)


    r = csv2latex("""					
LANCZOS			BICUBIC		
	SIFT kornia lanczos [0.25 0.25]		SIFT kornia bicubic [0. 0.]		
SIFT kornia lanczos [0. 0.]		SIFT kornia lanczos [-0.25 -0.25]		SIFT kornia bicubic [0.25 0.25]	SIFT kornia bicubic [-0.25 -0.25]
					
1.518	1.137	1.211	1.362	1.237	1.248
2.557	2.663	8.823	2.653	2.577	1.478
1.255	1.857	1.103	1.141	1.619	0.918
0.58	0.842	0.786	0.665	0.866	0.723
1.021	1.086	0.942	0.998	1.513	0.562
6.931	7.585	12.865	6.819	7.812	4.929""")
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