# Clean up locally Bazel:
#bazel clean --expunge
##############################################################################
#doxygen Doxyfile
##############################################################################
#If you want to delete all your commit history but keep the code in its #current state, do the following:
#
#git checkout --orphan latest_branch && git add -A && git commit -am "commit message" && git branch -D master && git branch -m master && git push -f origin master
###############################################################################
# Compilation etc
#
###############################################################################
#
#bazel run --compilation_mode="dbg" --verbose_failures //unittests:tests
#
##############################################################################
load("//tools:workspace.bzl", "eigen3_hdf5_workspace")
eigen3_hdf5_workspace()

