# 查找当前目录下的所有源文件
# 并将名称保存到 DIR_LIB_SRCS 变量
aux_source_directory(. DIR_SRCS)

list(REMOVE_ITEM DIR_SRCS "./VectorizedSolver.cpp")

# 生成链接库
add_library (${moc_lib} ${DIR_SRCS})
target_link_libraries(${moc_lib} ${MOC_LIBS}   )
target_include_directories(${moc_lib} PUBLIC ${MOC_INCLUDE_DIR})
set( MOC_LIBS  ${MOC_LIBS} ${moc_lib} PARENT_SCOPE)