# Two-factor-Commodity-model
A Python Implementation 

This model is based on Eduardo Schwartz and James E. Smith July 2000 paper "Short-Term Variations and Long-Term
Dynamics in Commodity Prices". You can find all the math details in this paper https://sites.ualberta.ca/~jbb/files/Schwartz.pdf

The opmizer for Kalman filter I used is fmin_slsqp. Compared with matlab optimizer fmincon, it is not very stable and constantly reports singular matrix error. For better performance, I recommend calling the matlab API to Python.

 I am a math major so the codes may not look that professonal. I strongly welcome any recommendations whcih could improve the process. You can actually find the full implemention of this model in matlab from internet- this example fits your need if you only want to do it in Python.

Please contact wendi.zhu1991@gmail.com for any question.
