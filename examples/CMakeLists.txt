#==============================================================================
#  TESTING TARGET
#==============================================================================

# Custom "piotests" target (builds the test executables)
add_custom_target (examples)

#==============================================================================
#  INCLUDE SOURCE DIRECTORIES
#==============================================================================

if ("${PLATFORM}" STREQUAL "goldbach" )
    ADD_SUBDIRECTORY(f03)
elseif ("${PLATFORM}" STREQUAL "yellowstone" )
    ADD_SUBDIRECTORY(f03)
    ADD_SUBDIRECTORY(c)
else()
    ADD_SUBDIRECTORY(f03)
    ADD_SUBDIRECTORY(c)
endif()
