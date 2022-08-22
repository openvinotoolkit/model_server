load("@bazel_tools//tools/cpp:toolchain_utils.bzl", "find_cpp_toolchain")

def _combine_impl(ctx):
    cc_toolchain = find_cpp_toolchain(ctx)    

    target_list = []
    for dep_target in ctx.attr.deps:        
        # CcInfo, InstrumentedFilesInfo, OutputGroupInfo      
        cc_info_linker_inputs = dep_target[CcInfo].linking_context.linker_inputs

        target_dirname_list = []
        for linker_in in cc_info_linker_inputs.to_list():            
            for linker_in_lib in linker_in.libraries:                
                if linker_in_lib.pic_static_library != None:
                    target_list += [linker_in_lib.pic_static_library]                    
                if linker_in_lib.static_library != None:
                    target_list += [linker_in_lib.static_library]
    
    output = ctx.outputs.output
    if ctx.attr.genstatic:
        cp_command = ""       
        processed_list = []
        processed_path_list = []
        for dep in target_list:
            cp_command += "cp -a "+ dep.path +" "+ output.dirname + "/&&"
            processed = ctx.actions.declare_file(dep.basename)
            processed_list += [processed]
            processed_path_list += [dep.path]
        cp_command += "echo'starting to run shell'"
        processed_path_list += [output.path]
  
        ctx.actions.run_shell(
            outputs = processed_list,
            inputs = target_list,
            command = cp_command,
        )

        command = "cd {} && ar -x {} {}".format(
                output.dirname,
                "&& ar -x ".join([dep.basename for dep in target_list]),
                "&& ar -rc libauto.a *.o"
            )
        print("command = ", command)
        ctx.actions.run_shell(
            outputs = [output],
            inputs = processed_list,
            command = command,
        )
    else:
        command = "export PATH=$PATH:{} && {} -shared -fPIC -Wl,--whole-archive {} -Wl,--no-whole-archive -Wl,-soname -o {}".format (
            cc_toolchain.ld_executable,
            cc_toolchain.compiler_executable,
            "".join([dep.path for dep in target_list]),
            output.path)
        print("command = ", command)
        ctx.actions.run_shell(
            outputs = [output],
            inputs = target_list,
            command = command,
        )

my_cc_combine = rule(
    implementation = _combine_impl,
    attrs = {
        "_cc_toolchain": attr.label(default = Label("@bazel_tools//tools/cpp:current_cc_toolchain")),
        "genstatic": attr.bool(default = False),
        "deps": attr.label_list(),
        "output": attr.output()
    },
)
