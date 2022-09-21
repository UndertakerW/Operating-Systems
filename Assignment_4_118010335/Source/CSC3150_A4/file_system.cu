#include "file_system.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

__device__ __managed__ u32 gtime = 0;

// initialize the volume by setting all bytes to 0
__device__ void init_volume(FileSystem *fs) {
	for (int i = 0; i < fs->STORAGE_SIZE; i++)
		fs->volume[i] = 0;
}

// read a filename in the volume into a char*
__device__ void read_filename(FileSystem *fs, int addr, char *dest) {
	int offset = 0;
	while (fs->volume[addr + offset]) {
		dest[offset] = fs->volume[addr + offset];
		offset++;
	}
	dest[offset] = '\0';
}

// write a filename into the volume
__device__ void write_filename(FileSystem *fs, int addr, char *s) {
	int i = 0;
	while (s[i]) {
		fs->volume[addr + i] = s[i];
		i++;
	}
	fs->volume[addr + i] = '\0';
}

// read a word in the volume
__device__ uint32_t read_word(FileSystem *fs, int addr) {
	uint32_t result = 0;
	for (int i = 0; i < 4; i++)
		result += fs->volume[addr + i] << (24 - 8 * i);
	return result;
}

// write a word into the volume
__device__ void write_word(FileSystem *fs, int addr, uint32_t value) {
	for (int i = 0; i < 4; i++)
		fs->volume[addr + i] = value >> (24 - 8 * i);
}

// read a halfword in the volume
__device__ uint16_t read_halfword(FileSystem *fs, int addr) {
	uint16_t result = 0;
	for (int i = 0; i < 2; i++)
		result += fs->volume[addr + i] << (8 - 8 * i);
	return result;
}

// write a word into the volume
__device__ void write_halfword(FileSystem *fs, int addr, short value) {
	for (int i = 0; i < 2; i++)
		fs->volume[addr + i] = value >> (8 - 8 * i);
}

// update the bitmap
__device__ void update_bitmap(FileSystem *fs) {
	// update the superblock (bit map)
	int filled_bytes_num = fs->block_count / 8;
	// filled bytes = 0b 1111 1111
	for (int i = 0; i < filled_bytes_num; i++)
		fs->volume[i] = 0b11111111;
	// half-filled byte = 0b ???? ???? (could be 0)
	int half_filled_byte = 0;
	for (int i = 0; i < fs->block_count % 8; i++)
		half_filled_byte += 1 << (7 - i);
	fs->volume[filled_bytes_num] = half_filled_byte;
	// unfilled bytes = 0b 0000 0000
	for (int i = filled_bytes_num + 1; i < fs->SUPERBLOCK_SIZE; i++)
		fs->volume[i] = 0;
}

// compact the volume
__device__ int compact(FileSystem *fs, int frag_start, int frag_size) {
	int frag_end = frag_start + frag_size - 1;
	int move_start = (frag_end + 1) * fs->STORAGE_BLOCK_SIZE + fs->FILE_BASE_ADDRESS;
	int move_size = (fs->block_count - 1 - frag_end) * fs->STORAGE_BLOCK_SIZE;
	// move the subsequent data to fill up the fragment
	for (int i = 0; i < move_size; i++) {
		int from = move_start + i;
		int to = frag_start * fs->STORAGE_BLOCK_SIZE + fs->FILE_BASE_ADDRESS + i;
		fs->volume[to] = fs->volume[from];
	}
	// update the FCBs
	for (int i = 0; i <= fs->file_count; i++) {
		int fcb_addr = read_FCB_address(fs, i);
		if (fcb_addr > frag_start) {
			int fcb_addr_new = fcb_addr - frag_size;
			write_FCB_address(fs, i, fcb_addr_new);
		}
	}
	// update the block count
	fs->block_count -= frag_size;
	// update the bit map
	update_bitmap(fs);
}

// get the length of a string
__device__ int get_length(const char* ptr) {
	int length = 0;
	while (*ptr++)
		length++;
	length++;
	return length;
}

// search the FCBs for a given filename, return its address if found
__device__ int find_filename(FileSystem *fs, const char *filename) {
	int filename_length = get_length(filename);
	if (filename_length > 20) {
		printf("Error: filename \"%s\" is over %d characters\n", filename, fs->MAX_FILENAME_SIZE);
	}
	// search among the FCBs
	char *fcb_filename = (char *)malloc(20 * sizeof(char));
	for (int i = 0; i < fs->file_count; i++) {
		int base_addr = fs->SUPERBLOCK_SIZE + i * fs->FCB_SIZE;
		read_filename(fs, base_addr, fcb_filename);
		bool found = true;
		for (int i = 0; i < filename_length; i++) {
			if (fcb_filename[i] != filename[i]) {
				found = false;
				break;
			}
		}
		if (found)
			return i;
	}
	free(fcb_filename);
	// if not found
	return -1;
}

// update the FCB filename
__device__ void write_FCB_filename(FileSystem *fs, int fcb_num, char* filename) {
	int base_addr = fs->SUPERBLOCK_SIZE + fcb_num * fs->FCB_SIZE;
	write_filename(fs, base_addr, filename);
}

// update the FCB address
__device__ void write_FCB_address(FileSystem *fs, int fcb_num, uint16_t address) {
	write_halfword(fs, fs->SUPERBLOCK_SIZE + fcb_num * fs->FCB_SIZE + fs->MAX_FILENAME_SIZE, address);
}

// update the FCB size
__device__ void write_FCB_size(FileSystem *fs, int fcb_num, uint16_t size) {
	write_halfword(fs, fs->SUPERBLOCK_SIZE + fcb_num * fs->FCB_SIZE + fs->MAX_FILENAME_SIZE + sizeof(uint16_t), size);
}

// update the FCB modified time
__device__ void write_FCB_mod_time(FileSystem *fs, int fcb_num, uint32_t time) {
	write_word(fs, fs->SUPERBLOCK_SIZE + fcb_num * fs->FCB_SIZE + fs->MAX_FILENAME_SIZE + sizeof(uint32_t), time);
}

// read the FCB filename into a char*
__device__ void read_FCB_filename(FileSystem *fs, int fcb_num, char* dest) {
	int base_addr = fs->SUPERBLOCK_SIZE + fcb_num * fs->FCB_SIZE;
	read_filename(fs, base_addr, dest);
}

// read the FCB address
__device__ uint16_t read_FCB_address(FileSystem *fs, int fcb_num) {
	return read_halfword(fs, fs->SUPERBLOCK_SIZE + fcb_num * fs->FCB_SIZE + fs->MAX_FILENAME_SIZE);
}

// read the FCB size
__device__ uint16_t read_FCB_size(FileSystem *fs, int fcb_num) {
	return read_halfword(fs, fs->SUPERBLOCK_SIZE + fcb_num * fs->FCB_SIZE + fs->MAX_FILENAME_SIZE + sizeof(uint16_t));
}

// read the FCB modified time
__device__ uint32_t read_FCB_mod_time(FileSystem *fs, int fcb_num) {
	return read_word(fs, fs->SUPERBLOCK_SIZE + fcb_num * fs->FCB_SIZE + fs->MAX_FILENAME_SIZE + sizeof(uint32_t));
}

// initialize the file system
__device__ void fs_init(FileSystem *fs, uchar *volume, int SUPERBLOCK_SIZE,
	int FCB_SIZE, int FCB_ENTRIES, int VOLUME_SIZE,
	int STORAGE_BLOCK_SIZE, int MAX_FILENAME_SIZE,
	int MAX_FILE_NUM, int MAX_FILE_SIZE, int FILE_BASE_ADDRESS) {
	// init variables
	fs->volume = volume;
	fs->file_count = 0;
	fs->block_count = 0;

	// init constants
	fs->SUPERBLOCK_SIZE = SUPERBLOCK_SIZE;
	fs->FCB_SIZE = FCB_SIZE;
	fs->FCB_ENTRIES = FCB_ENTRIES;
	fs->STORAGE_SIZE = VOLUME_SIZE;
	fs->STORAGE_BLOCK_SIZE = STORAGE_BLOCK_SIZE;
	fs->MAX_FILENAME_SIZE = MAX_FILENAME_SIZE;
	fs->MAX_FILE_NUM = MAX_FILE_NUM;
	fs->MAX_FILE_SIZE = MAX_FILE_SIZE;
	fs->FILE_BASE_ADDRESS = FILE_BASE_ADDRESS;
	fs->MAX_BLOCK_NUM = MAX_FILE_SIZE / STORAGE_BLOCK_SIZE;

	// init volume
	init_volume(fs);
}

// open a file and return a file pointer
__device__ u32 fs_open(FileSystem *fs, char *s, int op) {
	// search the FCBs for the filename
	int fcb_num = find_filename(fs, s);
	// if the filename exists
	if (fcb_num != -1) {
		int file_addr = read_FCB_address(fs, fcb_num);
		return file_addr;
	}
	// else if the filename does not exist
	else if (fcb_num == -1) {
		// if operation is write, create a new file
		if (op == G_WRITE) {
			if (fs->file_count >= fs->MAX_FILE_NUM) {
				printf("Error: the number of files reaches %d\n", fs->MAX_FILE_NUM);
				return 0x80000000;
			}
			// update the FCB
			write_FCB_filename(fs, fs->file_count, s);
			write_FCB_address(fs, fs->file_count, fs->block_count);
			write_FCB_size(fs, fs->file_count, 0);
			write_FCB_mod_time(fs, fs->file_count, gtime);
			// increase gtime by 1
			gtime++;
			// increase file count by 1
			fs->file_count++;
			// return a pointer to the next free block
			return fs->block_count;
		}
		// if operation is read
		else if (op == G_READ) {
			printf("Error: file \"%s\" does not exist.\n", s);
			return 0x80000000;
		}
	}
}

// read the content of a file into the result buffer
__device__ void fs_read(FileSystem *fs, uchar *output, u32 size, u32 fp) {
	// if fp is over the boundary
	if (fp > fs->MAX_BLOCK_NUM)
		printf("Error: file address \"%x\" is over the boundary.\n", fp);
	// else if (fp + blocks to read) is over the boundary
	else if (fp + (size - 1) / fs->STORAGE_BLOCK_SIZE + 1 > fs->MAX_BLOCK_NUM)
		printf("Error: size to read \"%d\" is over the boundary.\n", size);
	// else if fp is in the boundary, read the data into the result buffer
	else {
		for (int i = 0; i < size; i++)
			output[i] = fs->volume[fs->FILE_BASE_ADDRESS + fp * fs->STORAGE_BLOCK_SIZE + i];
	}
}

// write the content of the input buffer into a file
__device__ u32 fs_write(FileSystem *fs, uchar* input, u32 size, u32 fp) {
	int blocks_occupied = (size - 1) / fs->STORAGE_BLOCK_SIZE + 1;
	// invalid case 1: if fp is over the boundary
	if (fp > fs->MAX_BLOCK_NUM) {
		printf("Error: file address \"%x\" is over the boundary.\n", fp);
		return -1;
	}
	// invalid case 2: if fp points to some free block other than the first one
	if (fp > fs->block_count) {
		printf("Error: file address \"%x\" is empty.\n", fp);
		return -1;
	}
	// invalid case 3: if size > the max size of a file
	if (size > fs->MAX_FILE_SIZE / fs->MAX_FILE_NUM) {
		printf("Error: file size \"%d\" is over the limit.\n", size);
		return -1;
	}
	// find the corresponding FCB
	int fcb_num = -1;
	for (int i = 0; i < fs->file_count; i++) {
		u32 file_addr = read_FCB_address(fs, i);
		if (file_addr == fp) {
			fcb_num = i;
			break;
		}
	}
	// invalid case 4: FCB does not exist
	if (fcb_num == -1) {
		printf("Error: FCB for fp \"%x\" does not exist.\n", fp);
		return -1;
	}
	// valid case 1: if fp points to the first free block, then this is an empty file
	if (fp == fs->block_count) {
		// invalid case 5: insufficient storage
		if (blocks_occupied > fs->MAX_BLOCK_NUM - fs->block_count) {
			printf("Error: insufficient storage.\n");
			return -1;
		}
		// directly write the data into the volume
		for (int i = 0; i < size; i++)
			fs->volume[fs->FILE_BASE_ADDRESS + fs->block_count * fs->STORAGE_BLOCK_SIZE + i] = input[i];
		// update the block count
		fs->block_count += blocks_occupied;
	}
	// if fp points to a occupied block, then this is a non-empty file
	else if (fp < fs->block_count) {
		int file_size = read_FCB_size(fs, fcb_num);
		int file_blocks_occupied = (file_size - 1) / fs->STORAGE_BLOCK_SIZE + 1;
		// valid case 2: if the blocks of the original file are enough
		if (blocks_occupied <= file_blocks_occupied) {
			// write the data into the volume
			for (int i = 0; i < size; i++)
				fs->volume[fs->FILE_BASE_ADDRESS + fp * fs->STORAGE_BLOCK_SIZE + i] = input[i];
			// detect and eliminate fragment
			int frag_size = file_blocks_occupied - blocks_occupied;
			if (frag_size > 0) {
				int frag_start = fp + blocks_occupied;
				compact(fs, frag_start, frag_size);
			}
		}
		// valid case 3: if the blocks of the original file are not enough
		else {
			// invalid case 5: insufficient storage
			if (blocks_occupied - file_blocks_occupied > fs->MAX_BLOCK_NUM - fs->block_count) {
				printf("Error: insufficient storage.\n");
				return -1;
			}
			// compact the volume
			compact(fs, fp, file_blocks_occupied);
			// update the FCB address
			write_FCB_address(fs, fcb_num, fs->block_count);
			// write the data into the volume
			for (int i = 0; i < size; i++)
				fs->volume[fs->FILE_BASE_ADDRESS + fs->block_count * fs->STORAGE_BLOCK_SIZE + i] = input[i];
			// update the block count
			fs->block_count += blocks_occupied;
		}
	}
	// update the FCB
	write_FCB_size(fs, fcb_num, size);
	write_FCB_mod_time(fs, fcb_num, gtime);
	// increase gtime by 1
	gtime++;
	// update the bit map
	update_bitmap(fs);
}

// list out the files
__device__ void fs_gsys(FileSystem *fs, int op) {
	bool *printed = (bool *)malloc(fs->file_count * sizeof(bool));
	char *filename = (char *)malloc(fs->MAX_FILENAME_SIZE * sizeof(char));
	for (int i = 0; i < fs->file_count; i++)
		printed[i] = false;
	// sort by modified time
	if (op == LS_D) {
		printf("===sort by modified time===\n");
		int index;
		int current_mod_time;
		int max_mod_time;
		for (int i = 0; i < fs->file_count; i++) {
			max_mod_time = -1;
			for (int j = 0; j < fs->file_count; j++) {
				if (!printed[j]) {
					current_mod_time = read_FCB_mod_time(fs, j);
					if (current_mod_time > max_mod_time) {
						max_mod_time = current_mod_time;
						index = j;
					}
				}
			}
			read_FCB_filename(fs, index, filename);
			printf("%s\n", filename);
			printed[index] = true;
		}
	}
	// sort by file size
	else if (op == LS_S) {
		printf("===sort by file size===\n");
		int index;
		int current_size;
		int max_size;
		for (int i = 0; i < fs->file_count; i++) {
			max_size = -1;
			for (int j = 0; j < fs->file_count; j++) {
				if (!printed[j]) {
					current_size = read_FCB_size(fs, j);
					if (current_size > max_size) {
						max_size = current_size;
						index = j;
					}
				}
			}
			read_FCB_filename(fs, index, filename);
			printf("%s %d\n", filename, read_FCB_size(fs, index));
			printed[index] = true;
		}
	}
	free(printed);
	free(filename);
}

// remove a file
__device__ void fs_gsys(FileSystem *fs, int op, char *s) {
	if (op == RM) {
		int fcb_num = find_filename(fs, s);
		//if the file exists
		if (fcb_num != -1) {
			// get the file size and number of blocks occupied
			int file_size = read_FCB_size(fs, fcb_num);
			int block_occupied = (file_size - 1) / fs->STORAGE_BLOCK_SIZE + 1;
			// get the file address
			int file_address = read_FCB_address(fs, fcb_num);
			// compact the volume
			compact(fs, file_address, block_occupied);
			// compact the FCBs
			for (int i = fcb_num; i < fs->file_count; i++) {
				for (int j = 0; j < fs->FCB_SIZE; j++)
					fs->volume[fs->SUPERBLOCK_SIZE + i * fs->FCB_SIZE + j] = fs->volume[fs->SUPERBLOCK_SIZE + (i + 1) * fs->FCB_SIZE + j];
			}
			// update the file count
			fs->file_count--;
			// update the bit map
			update_bitmap(fs);
		}
		//if the file does not exist
		else if (fcb_num == -1)
			printf("Error: file \"%s\" does not exist.\n", s);
	}
}