from setuptools import find_packages,setup

setup(
    name='MCQgenrator',
    version='0.0.1',
    author='Thank you',
    author_email='sunny.savita@ineuron.ai',
    install_requires=["huggingface_hub","transformerS","accelerate","bitsandbytes","langchain","streamlit","python-dotenv","PyPDF2","langchain_community"],
    packages=find_packages()
)