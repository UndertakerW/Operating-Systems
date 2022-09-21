#include "virtual_memory.h"
#include <cuda.h>
#include <cuda_runtime.h>

// initialize the inverted page table
__device__ void init_invert_page_table(VirtualMemory *vm) {
	for (int i = 0; i < vm->PAGE_ENTRIES; i++) {
		// invert_page_table[i] (from 0 to PAGE_ENTRIES - 1) stores valid-invalid bit (initialized as false)
		vm->invert_page_table[i] = 0x80000000; // invalid := MSB is 1
		// invert_page_table[i] (from PAGE_ENTRIES to 2 * PAGE_ENTRIES) stores page number
		vm->invert_page_table[i + vm->PAGE_ENTRIES] = i;
	}
}

// initialize the virtual memory
__device__ void vm_init(VirtualMemory *vm, uchar *buffer, uchar *storage,
	u32 *invert_page_table, int *pagefault_num_ptr,
	int PAGESIZE, int INVERT_PAGE_TABLE_SIZE,
	int PHYSICAL_MEM_SIZE, int STORAGE_SIZE,
	int PAGE_ENTRIES) {
	// init variables
	vm->buffer = buffer;
	vm->storage = storage;
	vm->invert_page_table = invert_page_table;
	vm->pagefault_num_ptr = pagefault_num_ptr;

	// init constants
	vm->PAGESIZE = PAGESIZE;
	vm->INVERT_PAGE_TABLE_SIZE = INVERT_PAGE_TABLE_SIZE;
	vm->PHYSICAL_MEM_SIZE = PHYSICAL_MEM_SIZE;
	vm->STORAGE_SIZE = STORAGE_SIZE;
	vm->PAGE_ENTRIES = PAGE_ENTRIES;

	// before first vm_write or vm_read
	init_invert_page_table(vm);
}

// swap out and swap in pages
__device__ void swap(VirtualMemory *vm, int in_page_number, int out_page_number, int frame_number, bool swap_out) {
	// move from data buffer to secondary memory
	if (swap_out) {
		for (int i = 0; i < vm->PAGESIZE; i++)
			vm->storage[out_page_number * vm->PAGESIZE + i] = vm->buffer[frame_number * vm->PAGESIZE + i];
	}
	// move from secondary memory to data buffer
	for (int i = 0; i < vm->PAGESIZE; i++)
		vm->buffer[frame_number * vm->PAGESIZE + i] = vm->storage[in_page_number * vm->PAGESIZE + i];
}

// get a frame number on page fault
__device__ int get_frame_number_on_page_fault(VirtualMemory *vm, u32 page_number) {
	// count the page fault
	(*vm->pagefault_num_ptr)++;
	int frame_number_to_swap = 0;
	/* Case 1: the table is not full */
	// search the inverted page table for space
	for (int i = 0; i < vm->PAGE_ENTRIES; i++) {
		// if there exists an invalid entry, just swap in
		if (vm->invert_page_table[i] == 0x80000000) {
			// swap in
			swap(vm, page_number, vm->invert_page_table[i + vm->PAGE_ENTRIES], i, false);
			// update the inverted page table
			vm->invert_page_table[i + vm->PAGE_ENTRIES] = page_number;
			// update the LRU status
			update_lru(vm, i);
			// return this frame number
			return i;
		}
		// find the LRU frame number
		// pick the least indexed entry to be the victim page in case of tie
		else if (vm->invert_page_table[i] >
			vm->invert_page_table[frame_number_to_swap])
			frame_number_to_swap = i;
	}
	/* Case 2: the table is full */
	// i.e. no invalid entry
	// swap out and swap in
	swap(vm, page_number, vm->invert_page_table[frame_number_to_swap + vm->PAGE_ENTRIES], frame_number_to_swap, true);
	// update the inverted page table
	vm->invert_page_table[frame_number_to_swap + vm->PAGE_ENTRIES] = page_number;
	// update the LRU status
	update_lru(vm, frame_number_to_swap);
	// return the frame number
	return frame_number_to_swap;
}

// update the LRU status
__device__ void update_lru(VirtualMemory *vm, int frame_number) {
	for (int i = 0; i < vm->PAGE_ENTRIES; i++) {
		// if valid, increase the LRU status by 1
		// if the LRU status == 0x7FFFFFFF, do not increase
		if (vm->invert_page_table[i] != 0x80000000 
			&& vm->invert_page_table[i] != 0x7FFFFFFF)
			vm->invert_page_table[i]++;
	}
	// reset the LRU status to 0
	vm->invert_page_table[frame_number] = 0;
}

// translate a logical address into a physical address
__device__ int get_phy_addr(VirtualMemory *vm, u32 logic_addr) {
	u32 page_offset = logic_addr % vm->PAGESIZE;
	u32 page_number = logic_addr / vm->PAGESIZE;
	int phy_addr;
	/* Case 1: PAGE HIT */
	// search the inverted page table
	for (int i = 0; i < vm->PAGE_ENTRIES; i++) {
		// if "page_number" is in the inverted page table
		if (vm->invert_page_table[i + vm->PAGE_ENTRIES] == page_number) {
			if (vm->invert_page_table[i] != 0x80000000) {
				// update the LRU status
				update_lru(vm, i);
				// return the physical address
				phy_addr = (i * vm->PAGESIZE) + (int)page_offset;
				return phy_addr;
			}
		}
	}
	/* Case 2: PAGE MISS */
	// swap in the desired page from the secondary memory
	int frame_number = get_frame_number_on_page_fault(vm, page_number);
	// return the physical address
	phy_addr = (frame_number * vm->PAGESIZE) + (int)page_offset;
	return phy_addr;
}

// check if the address is valid (in bound)
__device__ bool valid_addr(VirtualMemory *vm, u32 addr) {
	if (addr > (u32)vm->STORAGE_SIZE) {
		printf("Logical address out of bound!\n");
		return false;
	}
	return true;
}

// read single element from data buffer
__device__ uchar vm_read(VirtualMemory *vm, u32 addr) {
	if (!valid_addr(vm, addr))
		return NULL;
	int phy_addr = get_phy_addr(vm, addr);
	return vm->buffer[phy_addr];
}

// write value into data buffer
__device__ void vm_write(VirtualMemory *vm, u32 addr, uchar value) {
	if (!valid_addr(vm, addr))
		return;
	int phy_addr = get_phy_addr(vm, addr);
	vm->buffer[phy_addr] = value;
}

// load elements from data to result buffer
__device__ void vm_snapshot(VirtualMemory *vm, uchar *results, int offset, int input_size) {
	for (int i = 0; i < input_size; i++)
		results[i + offset] = vm_read(vm, i);
}