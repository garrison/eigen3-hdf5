# If ms is linked as a submodule, path_prefix is ms's directory
# within the workspace (e.g. "ms/"), and ms_repo_name 
#(e.g. "ms = ms") is the name of the local_repository rule (e.g. "@ms").

def eigen3_hdf5_workspace(path_prefix = "",eigen3_hdf5_repo_name = ""):
#

  native.new_http_archive(
    name = "hdf5",
    url = "http://www.hdfgroup.org/ftp/HDF5/current/src/hdf5-1.10.0-patch1.tar.gz",
    sha256 = "6e78cfe32a10e6e0629393cdfddf6cfa536571efdaf85f08e35326e1b4e9eff0",
    strip_prefix = "hdf5-1.10.0-patch1",
    build_file = "hdf5.BUILD",
)

  native.new_git_repository(
    name = "eigen3_hdf5",
    remote = "https://github.com/garrison/eigen3-hdf5.git",
    commit = "2c782414251e75a2de9b0441c349f5f18fe929a2",
    build_file = "eigen3-hdf5.BUILD",
)

  native.new_http_archive(
    name = "gtest",
    url = "https://github.com/google/googletest/archive/master.zip",
    sha256 = "06e106e05b34eec6822e9ab069fdb5c09f6566ac52e390b72e5668ab91a20ff6",
    build_file = "gtest.BUILD",
    strip_prefix = "googletest-master/googletest",
  )

  native.new_http_archive(
    name = "eigen",
    url = "http://bitbucket.org/eigen/eigen/get/3.3.2.zip",
    sha256 = "3bc49c581dc335eeffea570572443be66891135e02316f7bedf86d762e408761",
    build_file = "eigen.BUILD",
    strip_prefix = "eigen-eigen-da9b4e14c255",
  )

  native.new_http_archive(
    name = "zlib",
    url = "http://zlib.net/zlib-1.2.11.tar.gz",
    sha256 = "c3e5e9fdd5004dcb542feda5ee4f0ff0744628baf8ed2dd5d66f8ca1197cb1a1",
    strip_prefix = "zlib-1.2.11",
    build_file = "zlib.BUILD",
  )