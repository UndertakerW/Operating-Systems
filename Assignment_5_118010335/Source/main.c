#include <linux/module.h>
#include <linux/moduleparam.h>
#include <linux/kernel.h>
#include <linux/init.h>
#include <linux/stat.h>
#include <linux/fs.h>
#include <linux/workqueue.h>
#include <linux/sched.h>
#include <linux/interrupt.h>
#include <linux/slab.h>
#include <linux/cdev.h>
#include <linux/delay.h>
#include <asm/uaccess.h>
#include "ioc_hw5.h"

MODULE_LICENSE("GPL");

#define PREFIX_TITLE "OS_AS5"


// DMA
#define DMA_BUFSIZE 64
#define DMASTUIDADDR 0x0        // Student ID
#define DMARWOKADDR 0x4         // RW function complete
#define DMAIOCOKADDR 0x8        // ioctl function complete
#define DMAIRQOKADDR 0xc        // ISR function complete
#define DMACOUNTADDR 0x10       // interrupt count function complete
#define DMAANSADDR 0x14         // Computation answer
#define DMAREADABLEADDR 0x18    // READABLE variable for synchronize
#define DMABLOCKADDR 0x1c       // Blocking or non-blocking IO
#define DMAOPCODEADDR 0x20      // data.a opcode
#define DMAOPERANDBADDR 0x21    // data.b operand1
#define DMAOPERANDCADDR 0x25    // data.c operand2
void *dma_buf;

// Declaration for file operations
static ssize_t drv_read(struct file *filp, char __user *buffer, size_t, loff_t*);
static int drv_open(struct inode*, struct file*);
static ssize_t drv_write(struct file *filp, const char __user *buffer, size_t, loff_t*);
static int drv_release(struct inode*, struct file*);
static long drv_ioctl(struct file *, unsigned int , unsigned long );

// cdev file_operations
static struct file_operations fops = {
      owner: THIS_MODULE,
      read: drv_read,
      write: drv_write,
      unlocked_ioctl: drv_ioctl,
      open: drv_open,
      release: drv_release,
};

// in and out function
void myoutc(unsigned char data,unsigned short int port);
void myouts(unsigned short data,unsigned short int port);
void myouti(unsigned int data,unsigned short int port);
unsigned char myinc(unsigned short int port);
unsigned short myins(unsigned short int port);
unsigned int myini(unsigned short int port);

// Work routine
static struct work_struct *work_routine;

// For input data structure
struct DataIn {
    char a;
    int b;
    short c;
} *dataIn;

// Device numbers
static int dev_major;
static int dev_minor;

// cdev structure pointer
static struct cdev *dev_cdevp;

// Device name
#define DEV_NAME "mydev"

// Device base minor
#define DEV_BASEMINOR 0

// Device count
#define DEV_COUNT 1

// Sleep time (single-shot)
#define SLEEP_TIME 1000

// Bonus: device name
#define IRQ_DEV_NAME "mydev_irq"

// Bonus: device structure
struct dev_t* dev_irq;

// Bonus: Interrupt Request number
// Share interrupt with i8042, the keyboard interrupt
const int IRQ_NUM = 1;

// Bonus: the count of keyboard interrupt requests
static int irq_count = 0;

// Arithmetic funciton
static void drv_arithmetic_routine(struct work_struct* ws);

// Utility function
// Check if a number is prime
bool is_prime(int n);
// Count a interrupt request
static irqreturn_t count_irq(int irq, void* dev_id);


// Input and output data from/to DMA
void myoutc(unsigned char data,unsigned short int port) {
    *(volatile unsigned char*)(dma_buf+port) = data;
}
void myouts(unsigned short data,unsigned short int port) {
    *(volatile unsigned short*)(dma_buf+port) = data;
}
void myouti(unsigned int data,unsigned short int port) {
    *(volatile unsigned int*)(dma_buf+port) = data;
}
unsigned char myinc(unsigned short int port) {
    return *(volatile unsigned char*)(dma_buf+port);
}
unsigned short myins(unsigned short int port) {
    return *(volatile unsigned short*)(dma_buf+port);
}
unsigned int myini(unsigned short int port) {
    return *(volatile unsigned int*)(dma_buf+port);
}


static int drv_open(struct inode* ii, struct file* ff) {
	try_module_get(THIS_MODULE);
    	printk("%s:%s(): device open\n", PREFIX_TITLE, __func__);
	return 0;
}

static int drv_release(struct inode* ii, struct file* ff) {
	module_put(THIS_MODULE);
    	printk("%s:%s(): device close\n", PREFIX_TITLE, __func__);
	return 0;
}

// The read operation of the device
static ssize_t drv_read(struct file *filp, char __user *buffer, size_t ss, loff_t* lo) {

	// If readable, read the answer
	if (myini(DMAREADABLEADDR) == 1){
		int ans;
		// Read the answer from DMA
		ans = myini(DMAANSADDR);
		// Print a kernel message
		printk("%s:%s(): ans = %d\n", PREFIX_TITLE, __func__, ans);
		// Transfer the answer to the user space
		put_user(ans, (int *) buffer);
		// Clean the result
		myouti(0, DMAANSADDR);
		// Set the DMA to be unreadable
		myouti(0, DMAREADABLEADDR);
		return 0;
	}
	else {
		printk("%s:%s(): DMA not readable\n", PREFIX_TITLE, __func__);
		return -1;
	}

}

// The write operation of the device
static ssize_t drv_write(struct file *filp, const char __user *buffer, size_t ss, loff_t* lo) {

	// Get the input data from the user space
	struct DataIn data;
	get_user(data.a, (char*) buffer);
	get_user(data.b, (int*) buffer+1);
	get_user(data.c, (int*) buffer+2);

	// Write the input data into the DMA
	myoutc(data.a, DMAOPCODEADDR);
	myouti(data.b, DMAOPERANDBADDR);
	myouti(data.c, DMAOPERANDCADDR);

	// Set the DMA to be unreadable
	myouti(0, DMAREADABLEADDR);

	// Initialize the work routine
	INIT_WORK(work_routine, drv_arithmetic_routine);

	// Blocking IO Step 1: Put the work into the system work queue
	// Non-blocking IO: Just put the work into the system work queue
	printk("%s:%s(): queue work\n", PREFIX_TITLE, __func__);
	schedule_work(work_routine);

	// Blocking IO Step 2: Flush works in the work queue
	if (myini(DMABLOCKADDR)) {
		printk("%s:%s(): block\n", PREFIX_TITLE, __func__);
		// Flush works on the work queue
		// Force execution of the kernel-global workqueue and block until its completion
		flush_scheduled_work();
		// Set the DMA to be readable
		myouti(1, DMAREADABLEADDR);
	}

	return 0;

}

// The ioctl setting of the device
static long drv_ioctl(struct file *filp, unsigned int cmd, unsigned long arg) {

	int dma_readable;
	// Get the argument from the user space
	int arg_value;
	get_user(arg_value, (int *)arg);
	switch (cmd) {
		// Write a student ID into the DMA
		// Print the student ID in the kernel log
		case HW5_IOCSETSTUID:
			myouti(arg_value, DMASTUIDADDR);
			printk("%s:%s(): My STUID is = %d\n", PREFIX_TITLE, __func__, arg_value);
			break;
		// Set DMA RWOK to be arg_value
		// Print OK in the kernel log if the R/W functions are completed
		case HW5_IOCSETRWOK:
			myouti(arg_value, DMARWOKADDR);
			printk("%s:%s(): RW OK\n", PREFIX_TITLE, __func__);
			break;
		// Set ioctlOK to be arg_value
		// Print OK in the kernel log if the ioctl funtction is completed
		case HW5_IOCSETIOCOK:
			myouti(arg_value, DMAIOCOKADDR);
			printk("%s:%s(): IOC OK\n", PREFIX_TITLE, __func__);
			break;
		// Set DMA IRQOK to be arg_value
		// Print OK in the kernel log if the bonus is completed
		case HW5_IOCSETIRQOK:
			myouti(arg_value, DMACOUNTADDR); // Count function OK
			myouti(arg_value, DMAIRQOKADDR); // Interrupt Service Routine OK
			printk("%s:%s(): IRQ OK\n", PREFIX_TITLE, __func__);
			break;
		// Set the write mode to be arg_value
		case HW5_IOCSETBLOCK:
			myouti(arg_value, DMABLOCKADDR);
			if (arg_value == 0)
				printk("%s:%s(): Non-blocking IO\n", PREFIX_TITLE, __func__);
			else if (arg_value == 1)
				printk("%s:%s(): Blocking IO\n", PREFIX_TITLE, __func__);
			else{
				printk("%s:%s(): invalid write function mode: %d\n", PREFIX_TITLE, __func__, arg_value);
				return -1;
			}
			break;
		// Wait until the DMA is readable
		// Used before a read operation when using non-blocking write mode
		case HW5_IOCWAITREADABLE:
			// Check if the DMA is readable
			dma_readable = myini(DMAREADABLEADDR);
			while (dma_readable != 1) {
				// Sleep for some time
				msleep(SLEEP_TIME);
				// Check if the DMA is readable again 
				dma_readable = myini(DMAREADABLEADDR);
			}
			// Write DMA readable into the user space
			put_user(dma_readable, (int*) arg);
			printk("%s:%s(): wait readable 1\n", PREFIX_TITLE, __func__);
			break;
		// Invalid command
		default:
			printk("%s:%s(): invalid command: %d\n", PREFIX_TITLE, __func__, cmd);
			return -1;
	}
	return 0;

}

// The arithmetic routine of the device
static void drv_arithmetic_routine(struct work_struct* ws) {

	int ans;
	int count;
	// Get the input data from the DMA
	struct DataIn data;
    data.a = myinc(DMAOPCODEADDR);
    data.b = myini(DMAOPERANDBADDR);
    data.c = myini(DMAOPERANDCADDR);

	// Perform an arithmetic operation on the input data
	ans = 0;
    switch(data.a) {
        case '+':
            ans = data.b + data.c;
            break;
        case '-':
            ans = data.b - data.c;
            break;
        case '*':
            ans = data.b * data.c;
            break;
        case '/':
            ans = data.b / data.c;
            break;
        case 'p':
			count = 0;
    		ans = data.b;
			if (ans < 0)
				ans = 0;
    		while (count < data.c) {
				ans++;
        		if(is_prime(ans))
            		count++;
    		}
            break;
        default:
            printk("%s:%s(): invalid operand: %c\n", PREFIX_TITLE, __func__, data.a);
			return;
    }

	printk("%s:%s(): %d %c %d = %d", PREFIX_TITLE, __func__, data.b, data.a, data.c, ans);
	// Write the answer into the DMA
	myouti(ans, DMAANSADDR);

	// Set the DMA to be readable if non-blocking I/O
	if (myini(DMABLOCKADDR) == 0)
		myouti(1, DMAREADABLEADDR);

}

// Check if a number is prime
bool is_prime(int n) {
	int i;
    if (n == 1) return false;
    if (n == 2) return true;
    if (n % 2 == 0) return false;
    for (i = 3; i <= n/2; i += 2)
        if (n % i == 0) return false;
    return true;
}

// Count the interrupt request if it is from the keyboard
// typedef irqreturn_t (*irq_handler_t)(int, void *);
static irqreturn_t count_irq(int irq, void* dev_id) {
	irq_count++;
	// return type: (*irq_handler_t)(int, void *)
	// IRQ_NONE = (0 << 0)
	// IRQ_HANDLED = (1 << 0)
	// IRQ_WAKE_THREAD = (1 << 1)
	return IRQ_NONE;
}

static int __init init_modules(void) {
    
	// local variable for this dev_t structure 
	dev_t dev;
	// temp variable for returned values
	int ret;

	printk("%s:%s():...............Start...............\n", PREFIX_TITLE, __func__);

	/* Register interrupt service routine */
	dev_irq = kzalloc(sizeof(typeof(dev_t)), GFP_KERNEL);
	ret = request_irq(IRQ_NUM, count_irq, IRQF_SHARED, IRQ_DEV_NAME, (void*) dev_irq);
	if (ret) {
		printk("%s:%s(): cannot request irq\n", PREFIX_TITLE, __func__);
		return ret;
	}
	printk("%s:%s(): request_irq %d return %d\n", PREFIX_TITLE, __func__, IRQ_NUM, ret);

	/* Register chrdev */ 
	// Allocate a range of char device numbers
	ret = alloc_chrdev_region(&dev, DEV_BASEMINOR, DEV_COUNT, DEV_NAME);
	// If alloc_chrdev_region() returns an error
	if (ret) {
		printk("%s:%s(): cannot alloc chrdev\n", PREFIX_TITLE, __func__);
		return ret;
	}
	// Get the major number
	dev_major = MAJOR(dev);
	// Get the first minor number
	dev_minor = MINOR(dev);
	printk("%s:%s(): register chrdev(%d, %d)\n", PREFIX_TITLE, __func__, dev_major, dev_minor);

	/* Init cdev and make it alive */
	// Allocate a cdev structure
	dev_cdevp = cdev_alloc();
	// Initialize the cdev, remembering fops
	cdev_init(dev_cdevp, &fops);
	dev_cdevp->owner = THIS_MODULE;
	// Add the device to the system, making it alive immediately
	ret = cdev_add(dev_cdevp, MKDEV(dev_major, dev_minor), DEV_COUNT);
	// If cdev_add() returns an error
	if (ret) {
		printk("%s:%s(): add chrdev failed\n", PREFIX_TITLE, __func__);
		return ret;
	}

	/* Allocate DMA buffer */
	printk("%s:%s(): allocate DMA buffer\n", PREFIX_TITLE, __func__);
	// Use kzalloc() to allocate and zero-set memory
	dma_buf = kzalloc(DMA_BUFSIZE, GFP_KERNEL);

	/* Allocate work routine */
	work_routine = kzalloc(sizeof(typeof(*work_routine)), GFP_KERNEL);

	return 0;
}

static void __exit exit_modules(void) {

	/* Free IRQ */
	free_irq(IRQ_NUM, (void*) dev_irq);
	kfree(dev_irq);
	printk("%s:%s(): interrupt count = %d\n", PREFIX_TITLE, __func__, irq_count);

	/* Free DMA buffer when exit modules */
	kfree(dma_buf);
	printk("%s:%s(): free DMA buffer\n", PREFIX_TITLE, __func__);

	/* Delete character device */
	unregister_chrdev_region(MKDEV(dev_major, dev_minor), DEV_COUNT);
	cdev_del(dev_cdevp);

	/* Free work routine */
	kfree(work_routine);
	printk("%s:%s(): unregister chrdev\n", PREFIX_TITLE, __func__);

	printk("%s:%s():..............End..............\n", PREFIX_TITLE, __func__);
}

module_init(init_modules);
module_exit(exit_modules);
