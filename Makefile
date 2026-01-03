CC     = gcc
CFLAGS = -O1 -Wall
LIBS   = -lm

OBJS = lenet_cnn_fixed.o fc.o pool.o conv.o utils.o

lenet_cnn_fixed: $(OBJS)
	$(CC) -o lenet_cnn_fixed $(OBJS) $(LIBS)

lenet_cnn_fixed.o: lenet_cnn_fixed.c lenet_cnn_fixed.h weights.h
	$(CC) $(CFLAGS) -c lenet_cnn_fixed.c

fc.o: fc.c lenet_cnn_fixed.h
	$(CC) $(CFLAGS) -c fc.c

pool.o: pool.c lenet_cnn_fixed.h
	$(CC) $(CFLAGS) -c pool.c

conv.o: conv.c lenet_cnn_fixed.h
	$(CC) $(CFLAGS) -c conv.c

utils.o: utils.c lenet_cnn_fixed.h
	$(CC) $(CFLAGS) -c utils.c

clean:
	rm -f $(OBJS) lenet_cnn_fixed

