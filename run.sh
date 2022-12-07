# torchrun --standalone --nproc_per_node=1 test_gemini.py -m_name bert -iter_num 1
torchrun --standalone --nproc_per_node=1 test_param_wrapper.py -m_name bert -iter_num 1
