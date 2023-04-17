from setuptools import setup


if __name__ == '__main__':
    with open("requirements.txt", 'r') as f:
        packages = f.readlines()

    setup(name='nn_executor',
          version='0.1.1',
          author='Michał Machura',
          author_email='michal.m.machura@gmail.com',
          description='Pytorch models serializer and deserializer - executor',
          python_requires='>=3.7, <4',
          package_dir={'': 'src'},
          download_url='https://github.com/MichalMachura/nn_executor',
          install_requires=packages,
          )