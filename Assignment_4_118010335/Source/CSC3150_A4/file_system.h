#ifndef VIRTUAL_MEMORY_H
#define VIRTUAL_MEMORY_H

#include <cuda.h>
#include <cuda_runtime.h>
#include <inttypes.h>

typedef unsigned char uchar;
typedef uint32_t u32;

#define G_WRITE 1
#define G_READ 0
#define LS_D 0
#define LS_S 1
#define RM 2

struct FileSystem {
	uchar *volume;
	int SUPERBLOCK_SIZE;
	int FCB_SIZE;
	int FCB_ENTRIES;
	int STORAGE_SIZE;
	int STORAGE_BLOCK_SIZE;
	int MAX_FILENAME_SIZE;
	int MAX_FILE_NUM;
	int MAX_FILE_SIZE;
	int FILE_BASE_ADDRESS;
	int MAX_BLOCK_NUM;

	int file_count;
	int block_count;
};

__device__ void fs_init(FileSystem *fs, uchar *volume, int SUPERBLOCK_SIZE,
	int FCB_SIZE, int FCB_ENTRIES, int VOLUME_SIZE,
	int STORAGE_BLOCK_SIZE, int MAX_FILENAME_SIZE,
	int MAX_FILE_NUM, int MAX_FILE_SIZE, int FILE_BASE_ADDRESS);

__device__ u32 fs_open(FileSystem *fs, char *s, int op);
__device__ void fs_read(FileSystem *fs, uchar *output, u32 size, u32 fp);
__device__ u32 fs_write(FileSystem *fs, uchar* input, u32 size, u32 fp);
__device__ void fs_gsys(FileSystem *fs, int op);
__device__ void fs_gsys(FileSystem *fs, int op, char *s);

__device__ void init_volume(FileSystem *fs);
__device__ void read_filename(FileSystem *fs, int addr, char *dest);
__device__ void write_filename(FileSystem *fs, int addr, char *s);
__device__ uint32_t read_word(FileSystem *fs, int addr);
__device__ void write_word(FileSystem *fs, int addr, uint32_t value);
__device__ uint16_t read_halfword(FileSystem *fs, int addr);
__device__ void write_halfword(FileSystem *fs, int addr, short value);
__device__ void update_bitmap(FileSystem *fs);
__device__ int compact(FileSystem *fs, int frag_start, int frag_size);
__device__ int get_length(const char* ptr);
__device__ int find_filename(FileSystem *fs, const char *filename);
__device__ void write_FCB_filename(FileSystem *fs, int fcb_num, char* filename);
__device__ void write_FCB_address(FileSystem *fs, int fcb_num, uint16_t address);
__device__ void write_FCB_size(FileSystem *fs, int fcb_num, uint16_t size);
__device__ void write_FCB_mod_time(FileSystem *fs, int fcb_num, uint32_t time);
__device__ void read_FCB_filename(FileSystem *fs, int fcb_num, char* dest);
__device__ uint16_t read_FCB_address(FileSystem *fs, int fcb_num);
__device__ uint16_t read_FCB_size(FileSystem *fs, int fcb_num);
__device__ uint32_t read_FCB_mod_time(FileSystem *fs, int fcb_num);

#endif
