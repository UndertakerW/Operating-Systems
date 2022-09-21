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

#define MKDIR 3
#define CD 4
#define CD_P 5
#define RM_RF 6
#define PWD 7

#define FILE_TYPE 0
#define DIR_TYPE 1

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

	int MAX_DIR_DEPTH;
	int MAX_DIR_FILE_NUM;

	int file_count;
	int block_count;

	int* current_dir;
};

__device__ void fs_init(FileSystem *fs, uchar *volume, int SUPERBLOCK_SIZE,
	int FCB_SIZE, int FCB_ENTRIES, int VOLUME_SIZE,
	int STORAGE_BLOCK_SIZE, int MAX_FILENAME_SIZE,
	int MAX_FILE_NUM, int MAX_FILE_SIZE, int FILE_BASE_ADDRESS,
	int MAX_DIR_DEPTH, int MAX_DIR_FILE_NUM);

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
__device__ int compact_files(FileSystem *fs, int frag_start, int frag_size);
__device__ int get_length(const char* ptr);
__device__ int find_filename(FileSystem *fs, const char *filename, uint8_t type);
__device__ void write_FCB_filename(FileSystem *fs, int fcb_num, char* filename);
__device__ void write_FCB_address(FileSystem *fs, int fcb_num, uint16_t address);
__device__ void write_FCB_size(FileSystem *fs, int fcb_num, uint16_t size);
__device__ void write_FCB_mod_time(FileSystem *fs, int fcb_num, uint32_t time);
__device__ void read_FCB_filename(FileSystem *fs, int fcb_num, char* dest);
__device__ uint16_t read_FCB_address(FileSystem *fs, int fcb_num);
__device__ uint16_t read_FCB_size(FileSystem *fs, int fcb_num);
__device__ uint32_t read_FCB_mod_time(FileSystem *fs, int fcb_num);

__device__ int get_current_dir_index(FileSystem *fs);
__device__ int get_current_depth(FileSystem *fs);
__device__ void write_FCB_parent(FileSystem *fs, int fcb_num, uint16_t parent);
__device__ void write_FCB_depth(FileSystem *fs, int fcb_num, uint8_t depth);
__device__ void write_FCB_type(FileSystem *fs, int fcb_num, uint8_t type);
__device__ int16_t read_FCB_parent(FileSystem *fs, int fcb_num);
__device__ uint8_t read_FCB_depth(FileSystem *fs, int fcb_num);
__device__ uint8_t read_FCB_type(FileSystem *fs, int fcb_num);
__device__ int get_current_dir_file_count(FileSystem *fs);
__device__ void remove_file_recursively(FileSystem *fs, int fcb_num, char *s);
__device__ void update_dir_size(FileSystem *fs, int dir_fcb_num, int delta_size);
__device__ void compact_FCBs(FileSystem *fs, int fcb_num);

#endif
