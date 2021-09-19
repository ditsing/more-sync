use std::ops::Deref;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::time::Duration;

/// A Carrier that manages the lifetime of an instance of type `T`.
///
/// The carrier owns the instance (the `target`). References to the `target` can
/// be obtained by calling the [`create_ref`](`Carrier::create_ref`) method. The
/// references returned by the method will be valid as long as the reference is
/// alive.
///
/// The carrier can be [*closed*](`Carrier::close`), after which no new
/// references can be obtained. The carrier can also [*wait*](`Carrier::wait`)
/// for all references it gave out to be dropped. The ownership of `target` will
/// be returned to the caller after the wait is complete. The caller can then
/// carry out clean-ups or any other type of work that requires an owned
/// instance of type `T`.
pub struct Carrier<T> {
    template: Arc<CarrierTarget<T>>,
    shutdown: AtomicBool,
}

impl<T> Carrier<T> {
    pub fn new(target: T) -> Self {
        Self {
            template: Arc::new(CarrierTarget {
                target,
                condvar: Default::default(),
                count: Mutex::new(0),
            }),
            shutdown: AtomicBool::new(false),
        }
    }

    pub fn ref_count(&self) -> usize {
        *self.template.lock_count()
    }

    pub fn create_ref(&self) -> Option<CarrierRef<T>> {
        if !self.shutdown.load(Ordering::Acquire) {
            Some(CarrierRef::new(&self.template))
        } else {
            None
        }
    }

    pub fn close(&self) {
        self.shutdown.store(true, Ordering::Release);
    }

    fn unwrap_or_panic(self) -> T {
        let arc = self.template;
        assert_eq!(Arc::strong_count(&arc), 1);

        match Arc::try_unwrap(arc) {
            Ok(t) => t.target,
            Err(_arc) => {
                panic!("The carrier should not have any outstanding references")
            }
        }
    }

    pub fn wait(self) -> T {
        {
            let count = self.template.lock_count();
            let count = self
                .template
                .condvar
                .wait_while(count, |count| *count != 0)
                .expect("The carrier lock should not be poisoned");

            assert_eq!(*count, 0);
        }
        self.unwrap_or_panic()
    }

    pub fn wait_timeout(self, timeout: Duration) -> Result<T, Self> {
        let count = {
            let count = self.template.lock_count();
            let (count, _result) = self
                .template
                .condvar
                .wait_timeout_while(count, timeout, |count| *count != 0)
                .expect("The carrier lock should not be poisoned");
            *count
        };

        if count == 0 {
            Ok(self.unwrap_or_panic())
        } else {
            Err(self)
        }
    }

    pub fn shutdown(self) -> T {
        self.close();
        self.wait()
    }

    pub fn shutdown_timeout(self, timeout: Duration) -> Result<T, Self> {
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

impl<T> CarrierTarget<T> {
    fn lock_count(&self) -> MutexGuard<usize> {
        self.count
            .lock()
            .expect("The carrier lock should not be poisoned")
    }
}

#[derive(Default)]
pub struct CarrierRef<T> {
    inner: Arc<CarrierTarget<T>>,
}

impl<T> CarrierRef<T> {
    fn new(inner: &Arc<CarrierTarget<T>>) -> Self {
        let mut count = inner.lock_count();
        *count += 1;

        CarrierRef {
            inner: inner.clone(),
        }
    }

    fn delete(&self) {
        let mut count = self.inner.lock_count();
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

impl<T> Deref for CarrierRef<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.inner.deref().target
    }
}

impl<T> Drop for CarrierRef<T> {
    fn drop(&mut self) {
        self.delete()
    }
}
