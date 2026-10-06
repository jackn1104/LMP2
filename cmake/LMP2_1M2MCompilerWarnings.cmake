function(lmp2_1m2m_set_project_warnings target)
  set(warnings_are_errors "")
  if(LMP2_1M2M_WARNINGS_AS_ERRORS)
    set(warnings_are_errors -Werror)
  endif()

  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
    target_compile_options(
      ${target}
      PRIVATE
        -Wall
        -Wextra
        -Wpedantic
        -Wconversion
        -Wshadow
        -Wnon-virtual-dtor
        ${warnings_are_errors}
    )
  elseif(MSVC)
    target_compile_options(${target} PRIVATE /W4)
    if(LMP2_1M2M_WARNINGS_AS_ERRORS)
      target_compile_options(${target} PRIVATE /WX)
    endif()
  endif()
endfunction()
