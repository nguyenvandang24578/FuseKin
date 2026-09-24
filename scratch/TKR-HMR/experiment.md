**H36M MPJPE (mm)     >> tot: 69.64**



H36M PA-MPJPE (mm)  >> tot: 48.45



MPVPE (mm)          >> tot: 84.35



H36M ACCEL (mm/s^2) >> tot: 6.86







H36M MPJPE (mm)     >> tot: 70.48



H36M PA-MPJPE (mm)  >> tot: 48.72



MPVPE (mm)          >> tot: 84.38



H36M ACCEL (mm/s^2) >> tot: 7.01





H36M MPJPE (mm)     >> tot: 70.67



H36M PA-MPJPE (mm)  >> tot: 48.57



**MPVPE (mm)          >> tot: 84.12**



H36M ACCEL (mm/s^2) >> tot: 7.13





weight\_path: './experiment/exp\_04-11\_15\_50/checkpoint/best.pth.tar' || FULL



H36M MPJPE (mm)     >> tot: 70.08



H36M PA-MPJPE (mm)  >> tot: 47.51



MPVPE (mm)          >> tot: 83.96



H36M ACCEL (mm/s^2) >> tot: 6.61



Fetch model weight from ./experiment/exp\_04-12\_02\_23/checkpoint/best.pth.tar || vẫn có joint3D và có KTA không có motion
H36M MPJPE (mm)     >> tot: 70.51



H36M PA-MPJPE (mm)  >> tot: 47.96



MPVPE (mm)          >> tot: 84.57



H36M ACCEL (mm/s^2) >> tot: 6.47





Fetch model weight from ./experiment/exp\_04-12\_11\_31/checkpoint/best.pth.tar || chỉ có joint3D và ko có KTA không có motion

H36M MPJPE (mm)     >> tot: 70.10



H36M PA-MPJPE (mm)  >> tot: 47.67



MPVPE (mm)          >> tot: 84.13



H36M ACCEL (mm/s^2) >> tot: 6.54

&#x20;Fetch model weight from ./experiment/exp\_04-12\_23\_27/checkpoint/best.pth.tar ||ko có KTA và có motion



Evaluation start...



H36M MPJPE (mm)     >> tot: 69.83



H36M PA-MPJPE (mm)  >> tot: 47.66



MPVPE (mm)          >> tot: 83.96



H36M ACCEL (mm/s^2) >> tot: 6.55



\------------------------------------

thay KTA = GCN Stactic đơn giản

H36M MPJPE (mm)     >> tot: 69.70



H36M PA-MPJPE (mm)  >> tot: 47.23



MPVPE (mm)          >> tot: 83.83



H36M ACCEL (mm/s^2) >> tot: 6.52

\----------------------ARTS---------------------------

H36M MPJPE (mm)     >> tot: 70.08



H36M PA-MPJPE (mm)  >> tot: 47.82



MPVPE (mm)          >> tot: 84.45



**H36M ACCEL (mm/s^2) >> tot: 6.48**



\----------------GCN+ATTN--------------------- 3 layers



H36M MPJPE (mm)     >> tot: 69.45



H36M PA-MPJPE (mm)  >> tot: 47.65



MPVPE (mm)          >> tot: 83.78



H36M ACCEL (mm/s^2) >> tot: 6.56



\----------------GCN+ATTN--------------------- 2 layers





**H36M MPJPE (mm)     >> tot: 69.12**



H36M PA-MPJPE (mm)  >> tot: 47.46



MPVPE (mm)          >> tot: 83.65



H36M ACCEL (mm/s^2) >> tot: 6.57

\----------------GCN+ATTN--------------------- 2 layers + giảm joint loss



H36M MPJPE (mm)     >> tot: 69.52



H36M PA-MPJPE (mm)  >> tot: 47.47



MPVPE (mm)          >> tot: 83.91



H36M ACCEL (mm/s^2) >> tot: 6.52



\----------------------------------- 2 layers +



&#x20;   lr: 0.00005

&#x20;   lr\_step: \[2, 8, 15, 23]

&#x20;   lr\_factor: 0.6



H36M MPJPE (mm)     >> tot: 69.39



**H36M PA-MPJPE (mm)  >> tot: 47.17**



**MPVPE (mm)          >> tot: 83.32**



H36M ACCEL (mm/s^2) >> tot: 6.54

