add_library(RTNeural STATIC
    activation/activation.h
 
    Model.h
    Layer.h
    dense/dense.h
    lstm/lstm.h

    model_loader.h
    RTNeural.h
    RTNeural.cpp
)

set_property(TARGET RTNeural PROPERTY POSITION_INDEPENDENT_CODE ON)
set_target_properties(RTNeural PROPERTIES LINKER_LANGUAGE CXX)
target_include_directories(RTNeural
    PUBLIC
        ../modules/json
    INTERFACE
        ..
)
