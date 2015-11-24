#==============================================================================
#  DEFINE THE TARGETS AND TESTS
#==============================================================================

add_executable (pioperf EXCLUDE_FROM_ALL
                pioperformance.F90)
add_executable (perftest EXCLUDE_FROM_ALL
                perftest.c)
target_link_libraries (pioperf piof)
target_link_libraries (perftest pioc lmpe mpe pthread m)
add_dependencies (tests pioperf perftest)

if ("${CMAKE_Fortran_COMPILER_ID}" STREQUAL "GNU")
    target_compile_options (pioperf
        PRIVATE -ffree-line-length-none)
endif()

if (CMAKE_Fortran_COMPILER_ID STREQUAL "NAG")
   set ( CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -mismatch_all" )
#    target_compile_options (gptl
#        PRIVATE -mismatch_all)
endif ()

if (PIO_HDF5_LOGGING) 
    target_compile_definitions (pioperf 
        PUBLIC LOGGING)
endif ()

if (PIO_HDF5_LOGGING) 
    target_compile_definitions (perftest
        PUBLIC LOGGING)
endif ()
