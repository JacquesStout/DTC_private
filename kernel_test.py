from dipy.denoise.enhancement_kernel import EnhancementKernel

D33 = 1
D44 = 0.02
t = 1

k = EnhancementKernel(D33, D44, t)

print('hi')