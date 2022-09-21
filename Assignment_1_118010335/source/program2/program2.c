#include <linux/module.h>
#include <linux/sched.h>
#include <linux/pid.h>
#include <linux/kthread.h>
#include <linux/kernel.h>
#include <linux/err.h>
#include <linux/slab.h>
#include <linux/printk.h>
#include <linux/jiffies.h>
#include <linux/kmod.h>
#include <linux/fs.h>

MODULE_LICENSE("GPL");

/* Structures */

struct wait_opts {
	enum pid_type wo_type;
	int wo_flags;
	struct pid* wo_pid;
	struct siginfo __user* wo_info;
	int __user* wo_stat;
	struct rusage __user* wo_rusage;
	wait_queue_t child_wait;
	int notask_error;
};

/* Extern Function Prototypes */

extern long _do_fork(
	unsigned long clone_flags,
	unsigned long stack_start,
	unsigned long stack_size,
	int __user* parent_tidptr,
	int __user* child_tidptr,
	unsigned long tls);

extern int do_execve(
	struct filename* filename,
	const char __user* const __user* __argv,
	const char __user* const __user* __envp);

extern long do_wait(struct wait_opts* wo);

extern struct filename* getname(const char __user* filename);

/* Function Prototypes */

static int __init program2_init(void);

static void __exit program2_exit(void);

int my_fork(void* argc);

void my_wait(pid_t pid);

void my_info(int sig);

int my_exec(void);

EXPORT_SYMBOL_GPL(my_fork);

/* Global Variables */

static struct task_struct* task;

//implement fork function
int my_fork(void *argc){
	
	pid_t pid;

	//set default sigaction for current process
	int i;
	struct k_sigaction *k_action = &current->sighand->action[0];
	for(i=0;i<_NSIG;i++){
		k_action->sa.sa_handler = SIG_DFL;
		k_action->sa.sa_flags = 0;
		k_action->sa.sa_restorer = NULL;
		sigemptyset(&k_action->sa.sa_mask);
		k_action++;
	}
	
	/* fork a process using do_fork */
	pid = _do_fork(SIGCHLD, (unsigned long)&my_exec, 0, NULL, NULL, 0);
	printk("[program2] : The child process has pid = %d\n", pid);
	printk("[program2] : This is the parent process, pid = %d\n", (int)current->pid);

	/* execute a test program in child process */
	
	/* wait until child process terminates */
	my_wait(pid);

	return 0;
}

static int __init program2_init(void){

	printk("[program2] : module_init\n");
	
	/* write your code here */
	
	/* create a kernel thread to run my_fork */

	printk("[program2] : module_init create kthread start\n");

	//create and start a kthread
	task = kthread_create(&my_fork, NULL, "my_thread");
	if (!IS_ERR(task)) {
		printk("[program2] : module_init kthread start\n");
		wake_up_process(task);
	}
	
	return 0;
}

static void __exit program2_exit(void){
	printk("[program2] : module_exit\n");
}

//implement wait function
void my_wait(pid_t pid) {

	int status;
	int a;

	struct wait_opts wo;
	struct pid* wo_pid = NULL;
	enum pid_type type;
	type = PIDTYPE_PID;
	wo_pid = find_get_pid(pid);

	wo.wo_type = type;
	wo.wo_pid = wo_pid;
	wo.wo_flags = WEXITED | WUNTRACED;
	wo.wo_info = NULL;
	wo.wo_stat = (int __user*) & status;
	wo.wo_rusage = NULL;

	a = do_wait(&wo);

	//output child process exit status
	my_info(*wo.wo_stat);

	put_pid(wo_pid);

	return;
}

//identify signals
int my_WEXITSTATUS(int status) {
	return ((status & 0xff00) >> 8);
}

int my_WSTOPSIG(int status) {
	return (my_WEXITSTATUS(status));
}

int my_WTERMSIG(int status) {
	return (status & 0x7f);
}

int my_WIFEXITED(int status) {
	return (my_WTERMSIG(status) == 0);
}

int my_WIFSTOPPED(int status) {
	return ((status & 0xff) == 0x7f);
}

int my_WIFSIGNALED(int status) {
	return (((signed char)((status & 0x7f) + 1) >> 1) > 0);
}


//print information about the child process
void my_info(int status) {

	//normal
	if (my_WIFEXITED(status)) {
		printk("[program2] : child process exit normally\n");
		printk("[program2] : The return signal is 0\n");
	}

	//signaled
	else if (my_WIFSIGNALED(status)) {
		char* signal = NULL;
		char* description = NULL;
		switch (my_WTERMSIG(status)) {
		case SIGHUP:
			signal = "SIGHUP";
			description = "hang up";
			break;
		case SIGINT:
			signal = "SIGINT";
			description = "interrupt";
			break;
		case SIGQUIT:
			signal = "SIGQUIT";
			description = "quit";
			break;
		case SIGILL:
			signal = "SIGILL";
			description = "illegal";
			break;
		case SIGTRAP:
			signal = "SIGTRAP";
			description = "trap";
			break;
		case SIGABRT:
			signal = "SIGABRT";
			description = "abort";
			break;
		case SIGBUS:
			signal = "SIGBUS";
			description = "bus";
			break;
		case SIGFPE:
			signal = "SIGFPE";
			description = "floating point exception";
			break;
		case SIGKILL:
			signal = "SIGKILL";
			description = "kill";
			break;
		case SIGSEGV:
			signal = "SIGSEGV";
			description = "segmentation fault";
			break;
		case SIGPIPE:
			signal = "SIGPIPE";
			description = "pipe";
			break;
		case SIGALRM:
			signal = "SIGALRM";
			description = "alarm";
			break;
		case SIGTERM:
			signal = "SIGTERM";
			description = "terminate";
			break;
		default:
			break;
		}

		if (signal) {
			printk("[program2] : get %s signal\n", signal);
			printk("[program2] : child process has %s error\n", description);
		}
		else {
			printk("\n!!UNKNOWN SIGNAL!!\n");
		}

		printk("[program2] : The return signal is %d\n", my_WTERMSIG(status));
	}

	//stopped
	else if (my_WIFSTOPPED(status)) {
		if (my_WSTOPSIG(status) == SIGSTOP) {
			printk("[program2] : child process get SIGSTOP signal\n");
			printk("[program2] : child process stopped\n");
		}
		else {
			printk("\n!!UNKNOWN SIGNAL!!\n");
		}

		printk("[program2] : The return signal is %d\n", my_WSTOPSIG(status));
	}

	//continued
	else {
		printk("[program2] : child process continued\n");
	}

}

//implement execute function
int my_exec(void) {

	int result;

	//prepare the arguments for do_execve()
	//path[] needs to be changed when the directory / filename is changed
	const char path[] = "/home/seed/work/assignment1/source/program2/test";
	const char* const argv[] = { path, NULL, NULL };
	const char* const envp[] = { "HOME=/", "PATH=/sbin:/user/sbin:/bin:/usr/bin", NULL };

	struct filename* my_filename = getname(path);

	printk("[program2] : child process\n");

	//execute the program
	result = do_execve(my_filename, argv, envp);

	//check the result
	//if result == 0, return 0
	if (!result) {
		return 0;
	}
	//else, call do_exit()
	else {
		do_exit(result);
	}
}

module_init(program2_init);
module_exit(program2_exit);
