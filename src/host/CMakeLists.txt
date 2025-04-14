file(GLOB cpp_files "${CMAKE_CURRENT_LIST_DIR}/*.cpp")
file(GLOB c_files "${CMAKE_CURRENT_LIST_DIR}/*.c")

add_library(pimblas SHARED ${cpp_files} ${c_files} )



set(OUTPUT_FILE ${CMAKE_BINARY_DIR}/pimblas_init.h)

add_custom_command(
    OUTPUT ${OUTPUT_FILE} # output file
  	COMMAND /bin/sh ${GEHD_SCRIPT} ${PIMBLAS_DEFAULT_KERENEL_DIR}  > ${OUTPUT_FILE}
	COMMENT "Create file ${OUTPUT_FILE}" 
)


add_custom_target(generate_hd ALL
    DEPENDS ${OUTPUT_FILE} # 
)
add_dependencies(pimblas  generate_hd)

target_include_directories(pimblas PRIVATE
           ${CMAKE_CURRENT_SOURCE_DIR} 
		   ${CMAKE_CURRENT_SOURCE_DIR}/../../include
		   ${UPH}/include/dpu    #    ./dpu-pkg-config --cflags --libs dpu   
		   ${CMAKE_BINARY_DIR}  # for pimblas_init.h
		   ${CMAKE_CURRENT_SOURCE_DIR}/../share
		   )


link_directories("${UPH}/lib64")
set(PIMBLAS_LINK_LIBS "dpu")
if(LOGGING)
list(APPEND PIMBLAS_LINK_LIBS "spdlog::spdlog" )
endif()

target_link_libraries(pimblas PRIVATE ${PIMBLAS_LINK_LIBS})  # ./dpu-pkg-config --cflags --libs dpu 
install(TARGETS pimblas DESTINATION lib)

target_compile_options(pimblas PRIVATE "-mavx" "-mavx2")

set_target_properties(pimblas PROPERTIES INSTALL_RPATH "${LIBSTDCXX_DIR}:${CND_HOME}/lib:${LD_LIBRARY_PATH}")
set_target_properties(pimblas PROPERTIES BUILD_RPATH "${LIBSTDCXX_DIR}")

install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/../../include/pimblas.h"
        DESTINATION include) 


#set_target_properties(pimblas PROPERTIES
#    VERSION 1.0
#    SOVERSION 1
#)

