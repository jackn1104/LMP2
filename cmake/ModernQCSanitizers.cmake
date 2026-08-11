function(modernqc_enable_sanitizers target)
  if(NOT MODERNQC_ENABLE_SANITIZERS)
    return()
  endif()

  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
    if(NOT MODERNQC_SANITIZER_SET MATCHES "^[a-z,]+$")
      message(FATAL_ERROR "MODERNQC_SANITIZER_SET contains unsupported characters")
    endif()
    target_compile_options(
      ${target}
      PRIVATE
        "-fsanitize=${MODERNQC_SANITIZER_SET}"
        -fno-omit-frame-pointer
    )
    target_link_options(${target} PRIVATE "-fsanitize=${MODERNQC_SANITIZER_SET}")
  else()
    message(FATAL_ERROR "MODERNQC_ENABLE_SANITIZERS is currently supported only with GNU- or Clang-family compilers")
  endif()
endfunction()
