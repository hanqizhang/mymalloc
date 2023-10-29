/*
 * Summary: This implementation of malloc and related functions uses the
 * Segregated Fits strategy introduced on pp864-865 of CSAPP.
 *
 * More specifically, we maintain an array of pointers to free lists of
 * different sizes. This array is not stored as a global variable, but rather
 * at the beginning of the heap, right before the prologue. Each free list is
 * an explicit list, with each block having a predecessor pointer and successor
 * pointer following the block header.
 *
 * Allocated blocks do not store such pointers and therefore the entire heap can
 * only be traversed like an implicit list.
 *
 * The placement policy is a variant of First Fit, in that the first block in a
 * suitable free list is popped and allocated for a new payload. However, when
 * we insert a new free block into a free list, we make sure to maintain an
 * ascending order in terms of block size. As a result, this placement policy is
 * also similar to Best Fit.
 *
 * Coalescing is performed everytime the heap is extended or a block is freed.
 */
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <unistd.h>
#include <string.h>
#include <stdint.h>
#include "mm.h"
#include "memlib.h"

/* uncomment the following line when debugging using mm_check */
//#define DEBUG      TRUE
/* uncomment the following line when debugging in verbose mode */
//#define VERBOSE    TRUE

/* Basic macros, reference: CSAPP 9.9 p857 */
#define ALIGNMENT  __SIZEOF_POINTER__
#define ALIGN(size) ((((size) + (ALIGNMENT-1)) / (ALIGNMENT)) * (ALIGNMENT))

#define WSIZE      __SIZEOF_POINTER__       // word, size of header/footer
#define DSIZE      2*WSIZE                  // double word
#define CHUNKSIZE ((1<<12) + DSIZE)  // extend heap by how many bytes
#define INITSIZE  ((1<<7) + DSIZE)   // initialize how many bytes
#define LISTSIZE   16      // how many free lists we want
#define THRESHOLD  7       // threshold tuned for placement policy

#define MAX(x, y) ((x) > (y)? (x) : (y))

// pack size and allocation bit into header/footer
#define PACK(size, alloc) ((size) | (alloc))

// read and write a word at address p
#define GET(p) (*(unsigned long *)(p))
#define PUT(p, val) (*(unsigned long *)(p) = (unsigned long)(val))

// read the size and allocated bit from address p
#define GET_SIZE(p) (GET(p) & ~0x7)
#define GET_ALLOC(p) (GET(p) & 0x1)

// given block ptr bp, compute address of its header and footer
#define HDRP(bp) ((char *)(bp) - WSIZE)
#define FTRP(bp) ((char *)(bp) + GET_SIZE(HDRP(bp)) - DSIZE)

// given block ptr bp, compute address of next and previous blocks
#define NEXT_BLKP(bp) ((char *)(bp) + GET_SIZE(((char *)(bp) - WSIZE)))
#define PREV_BLKP(bp) ((char *)(bp) - GET_SIZE(((char *)(bp) - DSIZE)))

// our free blocks have two words following the header,
// one for predecessor pointer and one for successor pointer
#define PRED_BLKP(bp) (*(char **)(bp))              // address of predecessor blk
#define SUCC_BLKP(bp) (*(char **)((char *)(bp) + WSIZE)) // addr of successor blk


/* Global variable */
static char * heap_ptr; // points to the prologue block of the heap


// we store pointers to free lists before the prologue block
// we can quickly get the address of any of the pointers
static inline char * freelists(int index) {
    return heap_ptr - (LISTSIZE + 1 - index) * WSIZE;
}

/* Helper function declarations */
static void * extend_heap(size_t size);
static void * coalesce(void * ptr);
static void * place(void * ptr, size_t size);
static void add_free(void * ptr, size_t size); // add free block to a free list
static void pop_free(void * ptr);              // delete free block from a list
static size_t align_size(size_t size);
#ifdef DEBUG
static int mm_check();                         // heap consistency checker
#endif


/*
 * Initilize the heap, including the free list pointers
 */
int mm_init(void)
{
    // create initial empty heap
#ifdef DEBUG
    mem_init();
#endif
    if ((heap_ptr = mem_sbrk((LISTSIZE + 4) * WSIZE)) == (void *) -1)
        return -1;

    // alignment padding is not needed when WSIZE % 8, but makes code compatible
    // with WSIZE % 4 but ALIGNMENT % 8 (e.g. 32 bit version, doubleword-aligned)
    PUT(heap_ptr, 0);
    PUT(heap_ptr + ((LISTSIZE + 1)*WSIZE), PACK(DSIZE, 1));  // prologue header
    PUT(heap_ptr + ((LISTSIZE + 2)*WSIZE), PACK(DSIZE, 1));  // prologue footer
    PUT(heap_ptr + ((LISTSIZE + 3)*WSIZE), PACK(0, 1));      // epilogue header

    for (int i = 1; i < LISTSIZE + 1; ++i) {
        PUT(heap_ptr + (i*WSIZE), 0);                        // free list pointers
    }
    heap_ptr += (LISTSIZE + 2) * WSIZE;

    // extend heap with a free block of CHUNKSIZE bytes
    if (extend_heap(INITSIZE) == NULL)
        return -1;

#ifdef VERBOSE
    printf("\n\n************* Heap initialized *************\n\n");
#endif
#ifdef DEBUG
    if (mm_check() == 0)
        return -1;
#endif

    return 0;
}

/*
 * Allocate memory for payload of size bytes
 */
void * mm_malloc(size_t size)
{
    if (size == 0) return NULL;

    // since we include predecessor and successor pointers in a free block
    // minimum block size is 4 words
    size = align_size(size);

    // look for a fitting size from free lists
    // and since we order within each free list from small to larger size blocks,
    // we just need to check the block pointed from the free list pointer
    // which is at the beginning of the heap (before the prologue)
    int index = 0;
    void * bp = NULL;
    while (index < LISTSIZE) {
        if (GET(freelists(index)) != 0 &&
            GET_SIZE(HDRP(*(char**)freelists(index))) > size) {
            bp = *(char **)freelists(index);
            break;
        }
        ++index;
    }

    // if no free block is found
    if (!bp) {
        if ((bp = extend_heap(MAX(size, CHUNKSIZE))) == NULL)
            return NULL;
    }
    // allocate new block in the free block we found or extended
    bp = place(bp, size);

#ifdef VERBOSE
    printf("Malloc'd for %lu bytes...\n", size);
#endif
#ifdef DEBUG
    mm_check();
#endif

    return bp;
}

/*
 * Free memory block, update free list, and coalesce
 */
void mm_free(void * bp)
{
    size_t size = GET_SIZE(HDRP(bp));
    PUT(HDRP(bp), PACK(size, 0));
    PUT(FTRP(bp), PACK(size, 0));
    add_free(bp, size);
    coalesce(bp);

#ifdef VERBOSE
    printf("Freed %lu bytes at %p...\n", size, bp);
#endif
#ifdef DEBUG
    mm_check();
#endif

}

/*
 * Reallocate memory for payload of size bytes, given a block
 */
void * mm_realloc(void * bp, size_t size)
{
    if (size == 0) return NULL;
    size = align_size(size);

    // case 0: return bp directly if size is less than size of bp
    if (GET_SIZE(HDRP(bp)) >= size)
        return bp;

    int rem_size;
    int next_epi  = !GET_SIZE(HDRP(NEXT_BLKP(bp)));
    int next_free = !GET_ALLOC(HDRP(NEXT_BLKP(bp)));
    void * new_bp = bp;

    // case 1: next blocks are usable
    if (next_free || next_epi) {
        rem_size = GET_SIZE(HDRP(bp)) + GET_SIZE(HDRP(NEXT_BLKP(bp))) - size;
        // case 1-a: next blocks usable, but not enough
        if (rem_size < 0) {
            if (extend_heap(MAX(CHUNKSIZE, -rem_size)) == NULL)
                return NULL;
            rem_size += MAX(CHUNKSIZE, -rem_size);
        }
        // case 1-b: next block usable, and sufficed or now suffices
        pop_free(NEXT_BLKP(bp));
        PUT(HDRP(bp), PACK(size + rem_size, 1));
        PUT(FTRP(bp), PACK(size + rem_size, 1));
    }

    // case 2: next blocks are not usable, call mm_maloc and free old block
    else {
        new_bp = mm_malloc(size);
        memcpy(new_bp, bp, GET_SIZE(HDRP(bp)));
        mm_free(bp);
    }

#ifdef VERBOSE
    printf("Realloc'd block at %p to %lu bytes...\n", bp, size);
#endif
#ifdef DEBUG
    mm_check();
#endif
    return new_bp;
}

...
