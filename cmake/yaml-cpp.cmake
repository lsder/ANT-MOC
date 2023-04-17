# 指定编译好的静态库文件的路径
set(yaml-cpp_ROOT ${CMAKE_BINARY_DIR}/thirdparty/yaml-cpp)

# 指定头文件所在的目录
find_path(ADD_INCLUDE_DIR yaml.h ${yaml-cpp_ROOT}/include/yaml-cpp)
find_library(ADD_LIBRARY yaml-cpp  ${yaml-cpp_ROOT}/lib64  )

if (ADD_INCLUDE_DIR AND ADD_LIBRARY)
    set(yaml-cpp_INCLUDE_DIR   ${yaml-cpp_ROOT}/include)
    set(yaml-cpp_LIB       ${yaml-cpp_ROOT}/lib64/libyaml-cpp.so)
else()
    include(ExternalProject)
    set(yaml-cpp_GIT_TAG  yaml-cpp-0.7.0)  # 指定版本
    set(yaml-cpp_GIT_URL      https://github.com/jbeder/yaml-cpp.git)  # 指定git仓库地址
    set(yaml-cpp_CONFIGURE    cd ${yaml-cpp_ROOT}/src/yaml-cpp && cmake -D CMAKE_INSTALL_PREFIX=${yaml-cpp_ROOT} -DYAML_BUILD_SHARED_LIBS=ON .)  # 指定配置指令（注意此处修改了安装目录，否则默认情况下回安装到系统目录）
    set(yaml-cpp_MAKE         cd ${yaml-cpp_ROOT}/src/yaml-cpp && make)  # 指定编译指令（需要覆盖默认指令，进入我们指定的yaml-cpp_ROOT目录下）
    set(yaml-cpp_INSTALL      cd ${yaml-cpp_ROOT}/src/yaml-cpp && make install)  # 指定安装指令（需要覆盖默认指令，进入我们指定的yaml-cpp_ROOT目录下）

    ExternalProject_Add(yaml-cpp
            PREFIX            ${yaml-cpp_ROOT}
            GIT_REPOSITORY    ${yaml-cpp_GIT_URL}
            GIT_TAG           ${yaml-cpp_GIT_TAG}
            CONFIGURE_COMMAND ${yaml-cpp_CONFIGURE}
            BUILD_COMMAND     ${yaml-cpp_MAKE}
            INSTALL_COMMAND   ${yaml-cpp_INSTALL}
    )
    set(yaml-cpp_INCLUDE_DIR   ${yaml-cpp_ROOT}/include)
    set(yaml-cpp_LIB       ${yaml-cpp_ROOT}/lib64/libyaml-cpp.so)
endif (ADD_INCLUDE_DIR AND ADD_LIBRARY)
