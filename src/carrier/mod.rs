use std::ops::Deref;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::time::Duration;

pub struct Carrier<T> {
    template: CarrierRef<T>,
    shutdown: AtomicBool,
}

impl<T> Carrier<T> {
    pub fn new(inner: T) -> Self {
        Self {
            template: CarrierRef {
                inner: Arc::new(CarrierTarget {
                    target: inner,
                    condvar: Default::default(),
                    count: Mutex::new(0),
                }),
            },
            shutdown: AtomicBool::new(false),
        }
    }

    pub fn ref_count(&self) -> usize {
        *self.template.lock_count()
    }

    pub fn create_ref(&self) -> Option<CarrierRef<T>> {
        if !self.shutdown.load(Ordering::Acquire) {
            Some(self.template.dup())
        } else {
            None
        }
    }

    pub fn close(&self) {
        self.shutdown.store(true, Ordering::Release);
    }

    pub fn wait(&self) {
        let count = self.template.lock_count();
        let _count = self
            .template
            .inner
            .condvar
            .wait_while(count, |count| *count != 0)
            .expect("The carrier lock should not be poisoned");
        assert_eq!(Arc::strong_count(&self.template.inner), 1);
    }

    pub fn wait_timeout(&self, timeout: Duration) -> bool {
        let count = self.template.lock_count();
        let (count, _result) = self
            .template
            .inner
            .condvar
            .wait_timeout_while(count, timeout, |count| *count != 0)
            .expect("The carrier lock should not be poisoned");
        return *count == 0;
    }

    pub fn shutdown(&self) {
        self.close();
        self.wait()
    }

    pub fn shutdown_timeout(&self, timeout: Duration) -> bool {
        self.close();
        self.wait_timeout(timeout)
    }
}

#[derive(Default)]
struct CarrierTarget<T> {
    target: T,

    condvar: Condvar,
    count: Mutex<usize>,
}

pub struct CarrierRef<T> {
    inner: Arc<CarrierTarget<T>>,
}

impl<T> CarrierRef<T> {
    fn lock_count(&self) -> MutexGuard<usize> {
        self.inner
            .count
            .lock()
            .expect("The carrier lock should not be poisoned")
    }

    fn dup(&self) -> Self {
        let mut count = self.lock_count();
        *count += 1;

        CarrierRef {
            inner: self.inner.clone(),
        }
    }

    fn dedup(&self) {
        let mut count = self.lock_count();
        *count -= 1;

        if *count == 0 {
            self.inner.condvar.notify_one();
        }
    }
}

impl<T> AsRef<T> for CarrierRef<T> {
    fn as_ref(&self) -> &T {
        &self.inner.as_ref().target
    }
}

impl<T> Default for CarrierRef<T>
where
    T: Default,
{
    fn default() -> Self {
        Self {
            inner: Default::default(),
        }
    }
}

impl<T> Deref for CarrierRef<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.inner.deref().target
    }
}

impl<T> Drop for CarrierRef<T> {
    fn drop(&mut self) {
        self.dedup()
    }
}
