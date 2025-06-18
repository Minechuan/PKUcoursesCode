#
# Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES, ALL RIGHTS RESERVED.
#
# This software product is a proprietary product of NVIDIA CORPORATION &
# AFFILIATES (the "Company") and all right, title, and interest in and to the
# software product, including all associated intellectual property rights, are
# and shall remain exclusively with the Company.
#
# This software product is governed by the End User License Agreement
# provided with the software product.
#

app_dependencies += dependency('glib-2.0')

cross_cpu_gpu_files = [
	APP_NAME + '_reducer.c',
]

app_srcs += [
	APP_NAME + '.c',
	APP_NAME + '_core.c',
	APP_NAME + '_daemon.c',
	APP_NAME + '_client.c',
	APP_NAME + '_ucx.c',
	APP_NAME + '_mem_pool.c',
	common_dir_path + '/utils.c'
]

executable(DOCA_PREFIX + APP_NAME, app_srcs + cross_cpu_gpu_files,
	c_args : base_c_args,
	dependencies : app_dependencies,
	include_directories : app_inc_dirs,
	install: install_apps)

if flag_enable_gpu_support
	tmp = []
	foreach f_name : cross_cpu_gpu_files
		tmp += (f_name + 'u')
	endforeach
	cross_cpu_gpu_files = tmp

	cuda_args = gpu_cuda_args + ['-Wno-sign-compare']

	executable(DOCA_PREFIX + APP_NAME + GPU_SUFFIX, app_srcs + cross_cpu_gpu_files,
		c_args : gpu_c_args,
		cuda_args : cuda_args,
		dependencies : app_dependencies + gpu_dependencies,
		include_directories : app_inc_dirs,
		install: install_apps)
endif
