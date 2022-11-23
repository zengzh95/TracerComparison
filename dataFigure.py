
import numpy as np
import matplotlib.pyplot as plt
import argparse


parser = argparse.ArgumentParser(description="Static Tracer")
parser.add_argument("-m_name", type=str, default="simplenet",
                    choices=["gpt2", "bert", "albert", "simplenet", "alexnet", "vgg16", "resnet18"],
                    help="model name")
args = parser.parse_args()

model_name = args.m_name
res_static = []
r_file1 = open("tracer_results/static_" + model_name + ".txt", "r")
for line in r_file1:
    line = line.strip()
    if line == "":
        continue
    res_static.append(float(line))
r_file1.close()

res_gemini = []
r_file2 = open("tracer_results/gemini_" + model_name + ".txt", "r")
for line in r_file2:
    line = line.strip()
    if line == "":
        continue
    res_gemini.append(float(line))
r_file2.close()

res_module_wrapper = []
r_file3 = open("tracer_results/module_wrapper_" + model_name + ".txt", "r")
for line in r_file3:
    line = line.strip()
    if line == "":
        continue
    res_module_wrapper.append(float(line))
r_file3.close()

res_verify = []
r_file4 = open("tracer_results/verify_" + model_name + ".txt", "r")
for line in r_file4:
    line = line.strip()
    if line == "":
        continue
    res_verify.append(float(line))
r_file4.close()

res_param_wrapper = []
r_file5 = open("tracer_results/param_wrapper_" + model_name + ".txt", "r")
for line in r_file5:
    line = line.strip()
    if line == "":
        continue
    res_param_wrapper.append(float(line))
r_file5.close()


res_static = np.array(res_static)
res_gemini = np.array(res_gemini)
res_module_wrapper = np.array(res_module_wrapper)
res_verify = np.array(res_verify)
res_param_wrapper = np.array(res_param_wrapper)

# res_gemini = res_gemini[:-1]

x1 = np.arange(0, len(res_static))
x2 = np.arange(0, len(res_gemini))
x3 = np.arange(0, len(res_module_wrapper))
x4 = np.arange(0, len(res_verify))
x5 = np.arange(0, len(res_param_wrapper))

l1 = plt.plot(x1, res_static, "r-", label="static")
l2 = plt.plot(x2, res_gemini, "b-", label="gemini")
l3 = plt.plot(x3, res_module_wrapper, "y-", label="moduleWra")
# l4 = plt.plot(x4, res_verify, "g--", label="verify")
l5 = plt.plot(x5, res_param_wrapper, "g-", label="paramWra")

plt.title(model_name)
plt.xlabel("Layers")
plt.ylabel("Peak Memory Size (MB)")
plt.legend()
plt.show()

