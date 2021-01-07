import threading
import Queue
import time
import random
import os.path as osp
import numpy as np
from PIL import Image
from ..utils.dataset_utils import parse_im_name
#ospj = osp.join
ospeu = osp.expanduser
# from TrainSet import Trainset.pre_process_im


class Counter(object):
    """A thread safe counter."""
    def __init__(self, val=0, max_val=0):
        self._value = val
        self.max_value = max_val
        self._lock = threading.Lock()

    def reset(self):
        with self._lock:
            self._value = 0

    def set_max_value(self, max_val):
        self.max_value = max_val

    def increment(self):
        with self._lock:
            if self._value < self.max_value:
                self._value += 1
                incremented = True
            else:
                incremented = False
            return incremented, self._value

    def get_value(self):
        with self._lock:
            return self._value


class Enqueuer(object):
    def __init__(self, get_element, num_elements, num_threads=1, queue_size=20):
        """
        Args:
          get_element: a function that takes a pointer and returns an element
          num_elements: total number of elements to put into the queue
          num_threads: num of parallel threads, >= 1
          queue_size: the maximum size of the queue. Set to some positive integer
            to save memory, otherwise, set to 0.
        """
        self.get_element = get_element
        assert num_threads > 0
        self.num_threads = num_threads
        self.queue_size = queue_size
        self.queue = Queue.Queue(maxsize=queue_size)
        # The pointer shared by threads.
        self.ptr = Counter(max_val=num_elements)
        # The event to wake up threads, it's set at the beginning of an epoch.
        # It's cleared after an epoch is enqueued or when the states are reset.
        self.event = threading.Event()
        # To reset states.
        self.reset_event = threading.Event()
        # The event to terminate the threads.
        self.stop_event = threading.Event()
        self.threads = []
        for _ in range(num_threads):
            thread = threading.Thread(target=self.enqueue)
            # Set the thread in daemon mode, so that the main program ends normally.
            thread.daemon = True
            thread.start()
            self.threads.append(thread)

    def start_ep(self):
        """Start enqueuing an epoch."""
        self.event.set()

    def end_ep(self):
        """When all elements are enqueued, let threads sleep to save resources."""
        self.event.clear()
        self.ptr.reset()
    def reset(self):
        """Reset the threads, pointer and the queue to initial states. In common
        case, this will not be called."""
        self.reset_event.set()
        self.event.clear()
        # wait for threads to pause. This is not an absolutely safe way. The safer
        # way is to check some flag inside a thread, not implemented yet.
        time.sleep(5)
        self.reset_event.clear()
        self.ptr.reset()
        self.queue = Queue.Queue(maxsize=self.queue_size)
    def set_num_elements(self, num_elements):
        """Reset the max number of elements."""
        self.reset()
        self.ptr.set_max_value(num_elements)
        '''
    #m_dict  = TrainSet.get_im_dict()
    im_list = list(im_dict.values())
    for _ in range(int(self.batch_size/self.images_per_id)):
      if self.ptr >= self.dataset_size:
        self.epoch_done = True
        break
      else:
        if self.id_ptr >= self.id_number:
          self.id_ptr = 0
        if len(im_list[self.id_ptr]) <  self.images_per_id:
          for i in range(self.image_per_id):
            im_name = random.sample(im_list[self.id_ptr],1)
            im_path = osp.join(self.im_dir, im_name)
            im = np.asarray(Image.open(im_path))
            im, mirrored = self.pre_process_im(im)
            id = parse_im_name(im_name, 'id')
            label = self.ids2labels[id]
            sample = (im,im_name,label,mirrored)
            samples.append(sample)
        else:
          im_names  = random.sample(im_list[self.id_ptr],self.images_per_id)
          for j in range(self.image_per_id):
            im_path = osp.join(self.im_dir, im_names[j])
            im = np.asarray(Image.open(im_path))
            im, mirrored = self.pre_process_im(im)
            id = parse_im_name(im_name[j], 'id')
            label = self.ids2labels[id]
            sample = (im, im_name[j], label, mirrored)
            samples.append(sample)
        self.id_ptr +=1
        self.ptr += self.images_per_id    self.ptr.set_max_value(num_elements)
      '''

    def stop(self):
        """Wait for threads to terminate."""
        self.stop_event.set()
        for thread in self.threads:
            thread.join()

    def enqueue(self):
        while not self.stop_event.isSet():
            # If the enqueuing event is not set, the thread just waits.
            if not self.event.wait(0.5):
                continue
            # Increment the counter to claim that this element has been enqueued by
            # this thread.
            incremented, ptr = self.ptr.increment()
            if incremented:
                element = self.get_element(ptr - 1)
                # When enqueuing, keep an eye on the stop and reset signal.
                while not self.stop_event.isSet() and not self.reset_event.isSet():
                    try:
                        # This operation will wait at most `timeout` for a free slot in
                        # the queue to be available.
                        self.queue.put(element, timeout=0.5)
                        break
                    except:
                        pass
            else:
                self.end_ep()
        print('Exiting thread {}!!!!!!!!'.format(
            threading.current_thread().name))


class Prefetcher(object):
    """This helper class enables sample enqueuing and batch dequeuing, to speed
    up batch fetching. It abstracts away the enqueuing and dequeuing logic."""

    def __init__(self, get_sample, pre_process_im, dataset_size, batch_size, final_batch=True,
                 num_threads=1, prefetch_size=200):
        """
        Args:
          get_sample: a function that takes a pointer (index) and returns a sample
          dataset_size: total number of samples in the dataset
          final_batch: True or False, whether to keep or drop the final incomplete
            batch
          num_threads: num of parallel threads, >= 1
          prefetch_size: the maximum size of the queue. Set to some positive integer
            to save memory, otherwise, set to 0.
        """
        self.full_dataset_size = dataset_size
        self.final_batch = final_batch
        final_sz = self.full_dataset_size % batch_size
        if not final_batch:
            dataset_size = self.full_dataset_size - final_sz
        self.dataset_size = dataset_size
        self.batch_size = batch_size
        #self.dataset = name
        self.enqueuer = Enqueuer(get_element=get_sample, num_elements=dataset_size,
                                 num_threads=num_threads, queue_size=prefetch_size)
        # The pointer indicating whether an epoch has been fetched from the queue
        #self.get_sample = get_sample
        self.pre_process_im = pre_process_im
        self.ptr = 0
        self.ep_done = True
#    self.id_number = 2533
        self.id_ptr = 0
        self.images_per_id = 8
        #self.im_dir = ospeu('/data2/reid-public/Dataset/pcb-format/{}/images'.format(self.dataset))

    def set_batch_size(self, batch_size):
        """You had better change batch size at the beginning of a new epoch."""
        final_sz = self.full_dataset_size % batch_size
        if not self.final_batch:
            self.dataset_size = self.full_dataset_size - final_sz
        self.enqueuer.set_num_elements(self.dataset_size)
        self.batch_size = batch_size
        self.ep_done = True

    def next_batch_test(self):
        """Return a batch of samples, meanwhile indicate whether the epoch is
        done. The purpose of this func is mainly to abstract away the loop and the
        boundary-checking logic.
        Returns:
          samples: a list of samples
          done: bool, whether the epoch is done
        """
        # Start enqueuing and other preparation at the beginning of an epoch.
        if self.ep_done:
            self.start_ep_prefetching()
        # Whether an epoch is done.
        self.ep_done = False
        samples = []
        for _ in range(self.batch_size):
            # Indeed, `>` will not occur.
            if self.ptr >= self.dataset_size:
                self.ep_done = True
                break
            else:
                self.ptr += 1
                sample = self.enqueuer.queue.get()
                # print('queue size {}'.format(self.enqueuer.queue.qsize()))
                samples.append(sample)
        # print 'queue size: {}'.format(self.enqueuer.queue.qsize())
        # Indeed, `>` will not occur.
        if self.ptr >= self.dataset_size:
            self.ep_done = True
        return samples, self.ep_done

    def next_batch(self, im_dict, ids2labels, im_dir):
        """Return a batch of samples, meanwhile indicate whether the epoch is
        done. The purpose of this func is mainly to abstract away the loop and the
        boundary-checking logic.
        Returns:
          samples: a list of samples
          done: bool, whether the epoch is done
        """
        # Start enqueuing and other preparation at the beginning of an epoch.
        if self.ep_done:
            self.start_ep_prefetching()
        # Whether an epoch is done.
        self.ep_done = False
        samples = []
        #im_dict  = TrainSet.get_im_dict()
        im_list = list(im_dict.values())
        for _ in range(int(self.batch_size/self.images_per_id)):
            if self.ptr >= self.dataset_size:
                self.epoch_done = True
                break
            else:
                if self.id_ptr >= len(im_dict.keys()):
                    self.id_ptr = 0
                if len(im_list[self.id_ptr]) < self.images_per_id:
                    for i in range(self.images_per_id):
                        im_name = random.sample(im_list[self.id_ptr], 1)
                        im_path = osp.join(im_dir, im_name[0])
                        # print im_dir
                        im = np.asarray(Image.open(im_path))
                        im, mirrored = self.pre_process_im(im)
                        id = parse_im_name(im_name[0], 'id')
                        label = ids2labels[id]
                        sample = (im, im_name[0], label, mirrored)
                        samples.append(sample)
                else:
                    im_names = random.sample(
                        im_list[self.id_ptr], self.images_per_id)
                    for j in range(self.images_per_id):
                        im_path = osp.join(im_dir, im_names[j])
                        # print "im_dir is :"
                        # print im_dir
                        im = np.asarray(Image.open(im_path))
                        im, mirrored = self.pre_process_im(im)
                        id = parse_im_name(im_names[j], 'id')
                        label = ids2labels[id]
                        #im,label,mirrored = self.get_sample(im_names[j])
                        sample = (im, im_names[j], label, mirrored)
                        samples.append(sample)
                self.id_ptr += 1
                self.ptr += self.images_per_id
        '''
    for _ in range(self.batch_size):
      # Indeed, `>` will not occur.
      '''
      if self.ptr >= self.dataset_size:
        self.ep_done = True
        break
      else:
        self.ptr += 1
        sample = self.enqueuer.queue.get()
        # print('queue size {}'.format(self.enqueuer.queue.qsize()))
        samples.append(sample)
    # print 'queue size: {}'.format(self.enqueuer.queue.qsize())
    # Indeed, `>` will not occur.
    '''
        if self.ptr >= self.dataset_size:
            self.ep_done = True
        return samples, self.ep_done

    def start_ep_prefetching(self):
        """
        NOTE: Has to be called at the start of every epoch.
        """
        self.enqueuer.start_ep()
        self.ptr = 0

    def stop(self):
        """This can be called to stop threads, e.g. after finishing using the
        dataset, or when existing the python main program."""
        self.enqueuer.stop()
