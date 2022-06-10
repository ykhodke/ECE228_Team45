The  main code can be run using "python project_main.py" 

The following modules were installed using Anaconda environment used in conjucation with CUDA and PyTorch

    # packages in environment at /home/ykhodke/anaconda3/envs/ECE228:
    #
    # Name                    Version                   Build  Channel
    _libgcc_mutex             0.1                        main  
    _openmp_mutex             5.1                       1_gnu  
    asttokens                 2.0.5              pyhd3eb1b0_0  
    backcall                  0.2.0              pyhd3eb1b0_0  
    blas                      1.0                         mkl  
    bottleneck                1.3.4            py39hce1f21e_0  
    brotli                    1.0.9                he6710b0_2  
    brotlipy                  0.7.0           py39h27cfd23_1003  
    bzip2                     1.0.8                h7b6447c_0  
    ca-certificates           2022.5.18.1          ha878542_0    conda-forge
    certifi                   2022.5.18.1      py39h06a4308_0  
    cffi                      1.15.0           py39hd667e15_1  
    charset-normalizer        2.0.4              pyhd3eb1b0_0  
    cloudpickle               2.0.0                    pypi_0    pypi
    cryptography              37.0.1           py39h9ce1e76_0  
    cudatoolkit               11.3.1               h2bc3f7f_2  
    cycler                    0.11.0             pyhd3eb1b0_0  
    cython                    0.29.30                  pypi_0    pypi
    dbus                      1.13.18              hb2f20db_0  
    debugpy                   1.5.1            py39h295c915_0  
    decorator                 5.1.1              pyhd3eb1b0_0  
    entrypoints               0.4              py39h06a4308_0  
    executing                 0.8.3              pyhd3eb1b0_0  
    expat                     2.4.4                h295c915_0  
    fasteners                 0.17.3                   pypi_0    pypi
    ffmpeg                    4.2.2                h20bf706_0  
    fontconfig                2.13.1               h6c09931_0  
    fonttools                 4.25.0             pyhd3eb1b0_0  
    freetype                  2.11.0               h70c0345_0  
    giflib                    5.2.1                h7b6447c_0  
    glfw                      2.5.3                    pypi_0    pypi
    glib                      2.69.1               h4ff587b_1  
    gmp                       6.2.1                h2531618_2  
    gnutls                    3.6.15               he1e5248_0  
    gst-plugins-base          1.14.0               h8213a91_2  
    gstreamer                 1.14.0               h28cd5cc_2  
    gym                       0.23.1                   pypi_0    pypi
    gym-notices               0.0.6                    pypi_0    pypi
    icu                       58.2                 he6710b0_3  
    idna                      3.3                pyhd3eb1b0_0  
    imageio                   2.19.2                   pypi_0    pypi
    importlib-metadata        4.11.3                   pypi_0    pypi
    intel-openmp              2021.4.0          h06a4308_3561  
    ipykernel                 6.9.1            py39h06a4308_0  
    ipython                   8.3.0            py39h06a4308_0  
    jedi                      0.18.1           py39h06a4308_1  
    jpeg                      9e                   h7f8727e_0  
    jupyter_client            7.2.2            py39h06a4308_0  
    jupyter_core              4.10.0           py39h06a4308_0  
    kiwisolver                1.4.2            py39h295c915_0  
    lame                      3.100                h7b6447c_0  
    lcms2                     2.12                 h3be6417_0  
    ld_impl_linux-64          2.38                 h1181459_1  
    libffi                    3.3                  he6710b0_2  
    libgcc-ng                 11.2.0               h1234567_0  
    libgfortran-ng            7.5.0               ha8ba4b0_17  
    libgfortran4              7.5.0               ha8ba4b0_17  
    libgomp                   11.2.0               h1234567_0  
    libidn2                   2.3.2                h7f8727e_0  
    libopus                   1.3.1                h7b6447c_0  
    libpng                    1.6.37               hbc83047_0  
    libsodium                 1.0.18               h7b6447c_0  
    libstdcxx-ng              11.2.0               h1234567_0  
    libtasn1                  4.16.0               h27cfd23_0  
    libtiff                   4.2.0                h2818925_1  
    libunistring              0.9.10               h27cfd23_0  
    libuuid                   1.0.3                h7f8727e_2  
    libuv                     1.40.0               h7b6447c_0  
    libvpx                    1.7.0                h439df22_0  
    libwebp                   1.2.2                h55f646e_0  
    libwebp-base              1.2.2                h7f8727e_0  
    libxcb                    1.15                 h7f8727e_0  
    libxml2                   2.9.12               h74e7548_2  
    lz4-c                     1.9.3                h295c915_1  
    matplotlib                3.5.1            py39h06a4308_1  
    matplotlib-base           3.5.1            py39ha18d171_1  
    matplotlib-inline         0.1.2              pyhd3eb1b0_2  
    mkl                       2021.4.0           h06a4308_640  
    mkl-service               2.4.0            py39h7f8727e_0  
    mkl_fft                   1.3.1            py39hd3c417c_0  
    mkl_random                1.2.2            py39h51133e4_0  
    mujoco-py                 2.1.2.14                 pypi_0    pypi
    munkres                   1.1.4                      py_0  
    ncurses                   6.3                  h7f8727e_2  
    nest-asyncio              1.5.5            py39h06a4308_0  
    nettle                    3.7.3                hbbd107a_1  
    numexpr                   2.8.1            py39h6abb31d_0  
    numpy                     1.22.3           py39he7a7128_0  
    numpy-base                1.22.3           py39hf524024_0  
    openh264                  2.1.1                h4ff587b_0  
    openssl                   1.1.1o               h7f8727e_0  
    packaging                 21.3               pyhd3eb1b0_0  
    pandas                    1.4.2            py39h295c915_0  
    parso                     0.8.3              pyhd3eb1b0_0  
    pcre                      8.45                 h295c915_0  
    pexpect                   4.8.0              pyhd3eb1b0_3  
    pickleshare               0.7.5           pyhd3eb1b0_1003  
    pillow                    9.0.1            py39h22f2fdc_0  
    pip                       21.2.4           py39h06a4308_0  
    prompt-toolkit            3.0.20             pyhd3eb1b0_0  
    ptyprocess                0.7.0              pyhd3eb1b0_2  
    pure_eval                 0.2.2              pyhd3eb1b0_0  
    pycparser                 2.21               pyhd3eb1b0_0  
    pydicom                   2.3.0              pyh6c4a22f_0    conda-forge
    pygame                    2.1.2                    pypi_0    pypi
    pygments                  2.11.2             pyhd3eb1b0_0  
    pyopenssl                 22.0.0             pyhd3eb1b0_0  
    pyparsing                 3.0.4              pyhd3eb1b0_0  
    pyqt                      5.9.2            py39h2531618_6  
    pysocks                   1.7.1            py39h06a4308_0  
    python                    3.9.12               h12debd9_0  
    python-dateutil           2.8.2              pyhd3eb1b0_0  
    python_abi                3.9                      2_cp39    conda-forge
    pytorch                   1.11.0          py3.9_cuda11.3_cudnn8.2.0_0    pytorch
    pytorch-mutex             1.0                        cuda    pytorch
    pytz                      2021.3             pyhd3eb1b0_0  
    pyzmq                     22.3.0           py39h295c915_2  
    qt                        5.9.7                h5867ecd_1  
    readline                  8.1.2                h7f8727e_1  
    requests                  2.27.1             pyhd3eb1b0_0  
    scipy                     1.7.3            py39hc147768_0    anaconda
    setuptools                61.2.0           py39h06a4308_0  
    sip                       4.19.13          py39h295c915_0  
    six                       1.16.0             pyhd3eb1b0_1  
    sqlite                    3.38.3               hc218d9a_0  
    stack_data                0.2.0              pyhd3eb1b0_0  
    tk                        8.6.11               h1ccaba5_1  
    torchaudio                0.11.0               py39_cu113    pytorch
    torchvision               0.12.0               py39_cu113    pytorch
    tornado                   6.1              py39h27cfd23_0  
    traitlets                 5.1.1              pyhd3eb1b0_0  
    typing_extensions         4.1.1              pyh06a4308_0  
    tzdata                    2022a                hda174b7_0  
    urllib3                   1.26.9           py39h06a4308_0  
    wcwidth                   0.2.5              pyhd3eb1b0_0  
    wheel                     0.37.1             pyhd3eb1b0_0  
    x264                      1!157.20191217       h7b6447c_0  
    xz                        5.2.5                h7f8727e_1  
    zeromq                    4.3.4                h2531618_0  
    zipp                      3.8.0                    pypi_0    pypi
    zlib                      1.2.12               h7f8727e_2  
    zstd                      1.5.2                ha4553b6_0  