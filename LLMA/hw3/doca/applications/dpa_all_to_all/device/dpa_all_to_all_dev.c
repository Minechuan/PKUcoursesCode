/*
 * Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES, ALL RIGHTS RESERVED.
 *
 * This software product is a proprietary product of NVIDIA CORPORATION &
 * AFFILIATES (the "Company") and all right, title, and interest in and to the
 * software product, including all associated intellectual property rights, are
 * and shall remain exclusively with the Company.
 *
 * This software product is governed by the End User License Agreement
 * provided with the software product.
 *
 */

#include <doca_dpa_dev.h>
#include <doca_dpa_dev_rdma.h>
#include <doca_dpa_dev_buf.h>
#include <doca_dpa_dev_sync_event.h>

#define SYNC_EVENT_MASK_FFS (0xFFFFFFFFFFFFFFFF) /* Mask for doca_dpa_dev_sync_event_wait_gt() wait value */

/*
 * Alltoall kernel function.
 * Performs RDMA write operations using doca_dpa_dev_rdma_write() from local buffer to remote buffer.
 *
 * @rdmas_dev_ptr [in]: An array of DOCA DPA RDMA handlers
 * @local_buf_addr [in]: local buffer address for alltoall
 * @local_buf_mmap_handle [in]: local buffer mmap handle for alltoall
 * @count [in]: Number of elements to write
 * @type_length [in]: Length of each element
 * @num_ranks [in]: Number of the MPI ranks
 * @my_rank [in]: The rank of the current process
 * @remote_recvbufs_dev_ptr [in]: Device pointer of array holding remote buffers addresses for alltoall
 * @remote_recvbufs_mmap_handles_dev_ptr [in]: Device pointer of array holding remote buffers mmap handle for alltoall
 * @local_events_dev_ptr [in]: Device pointer of DPA handles to communication events that will be updated by remote MPI
 * ranks
 * @remote_events_dev_ptr [in]: Device pointer of DPA handles to communication events on other nodes that will be
 * updated by this rank
 * @a2a_seq_num [in]: The number of times we called the alltoall_kernel in iterations
 */
__dpa_global__ void alltoall_kernel(doca_dpa_dev_uintptr_t rdmas_dev_ptr,
				    uint64_t local_buf_addr,
				    doca_dpa_dev_mmap_t local_buf_mmap_handle,
				    uint64_t count,
				    uint64_t type_length,
				    uint64_t num_ranks,
				    uint64_t my_rank,
				    doca_dpa_dev_uintptr_t remote_recvbufs_dev_ptr,
				    doca_dpa_dev_uintptr_t remote_recvbufs_mmap_handles_dev_ptr,
				    doca_dpa_dev_uintptr_t local_events_dev_ptr,
				    doca_dpa_dev_uintptr_t remote_events_dev_ptr,
				    uint64_t a2a_seq_num)
{
	/* Convert the RDMA DPA device pointer to rdma handle type */
	doca_dpa_dev_rdma_t *rdma_handles = (doca_dpa_dev_rdma_t *)rdmas_dev_ptr;
	/* Convert the remote receive buffer addresses DPA device pointer to array of pointers */
	uintptr_t *remote_recvbufs = (uintptr_t *)remote_recvbufs_dev_ptr;
	/* Convert the remote receive buffer mmap handles DPA device pointer to array of mmap handle type */
	doca_dpa_dev_mmap_t *remote_recvbufs_mmap_handles = (doca_dpa_dev_mmap_t *)remote_recvbufs_mmap_handles_dev_ptr;
	/* Convert the local events DPA device pointer to local events handle type */
	doca_dpa_dev_sync_event_t *local_events = (doca_dpa_dev_sync_event_t *)local_events_dev_ptr;
	/* Convert the remote events DPA device pointer to remote events handle type */
	doca_dpa_dev_sync_event_remote_net_t *remote_events =
		(doca_dpa_dev_sync_event_remote_net_t *)remote_events_dev_ptr;
	/* Get the rank of current thread that is running */
	unsigned int thread_rank = doca_dpa_dev_thread_rank();
	/* Get the number of all threads that are running this kernel */
	unsigned int num_threads = doca_dpa_dev_num_threads();
	unsigned int i;

	/*
	 * Each process should perform as the number of processes RDMA write operations with local and remote buffers
	 * according to the rank of the local process and the rank of the remote processes (we iterate over the rank
	 * of the remote process).
	 * Each process runs num_threads threads on this kernel so we divide the number RDMA write operations (which is
	 * the number of processes) by the number of threads.
	 */
	for (i = thread_rank; i < num_ranks; i += num_threads) {
		doca_dpa_dev_rdma_post_write(rdma_handles[i],
					     remote_recvbufs_mmap_handles[i],
					     remote_recvbufs[i] + (my_rank * count * type_length),
					     local_buf_mmap_handle,
					     local_buf_addr + (i * count * type_length),
					     (type_length * count),
					     1);

		doca_dpa_dev_rdma_signal_set(rdma_handles[i], remote_events[i], a2a_seq_num);
	}

	/*
	 * Each thread should wait on his local events to make sure that the
	 * remote processes have finished RDMA write operations.
	 * Each thread should also synchronize its rdma dpa handles to make sure
	 * that the local RDMA operation calls has finished
	 */
	for (i = thread_rank; i < num_ranks; i += num_threads) {
		doca_dpa_dev_sync_event_wait_gt(local_events[i], a2a_seq_num - 1, SYNC_EVENT_MASK_FFS);
		doca_dpa_dev_rdma_synchronize(rdma_handles[i]);
	}
}
