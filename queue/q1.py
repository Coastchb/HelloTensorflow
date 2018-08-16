import tensorflow as tf

# coordinator ?

def print_q(sess,q):
  qlen = sess.run(q.size())
  for i in range(qlen):
    print(sess.run(q.dequeue()))

with tf.Session() as sess:
  q = tf.FIFOQueue(4, 'float')
  init = q.enqueue_many(([1.0,2.0,3.0],))
      # Error: init = q.enqueue_many([1.0,2.0,3.0],)
      # Error: init = q.enqueue_many([1.0,2.0,3.0])

  init3 = q.enqueue([4.0])   # OK: init3 = q.enqueue(4.0)
  init2 = q.dequeue()

  sess.run(init)
  sess.run(init2)
  sess.run(init3)

  #print_q(sess,q);


  init4 = q.enqueue(5.0)
  init5 = q.enqueue(6.0)
  init6 = q.enqueue(7.0)
  init7 = q.dequeue();
  init8 = q.dequeue()
  init9 = q.dequeue()

  sess.run(init4)

  # will be blocked if run init5 or init6
  #sess.run(init5)
  #sess.run(init6)
  #sess.run(init7)
 # sess.run(init8)
 # sess.run(init9)

  print("After more operation:")
  print_q(sess, q)