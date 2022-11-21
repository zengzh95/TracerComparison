# TracerComparison

python3 test_static.py -m_name simplenet

python3 test_wrapper.py -m_name simplenet -iter_num 1

torchrun --standalone --nproc_per_node=1 test_gemini.py -m_name simplenet -iter_num 1

python3 dataFigure.py -m_name simplenet
