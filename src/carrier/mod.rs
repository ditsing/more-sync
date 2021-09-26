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
///
/// ```
/// use more_sync::Carrier;
///
/// // Create a carrier that holds a mutex.
/// let carrier = Carrier::new(std::sync::Mutex::new(7usize));
///
/// // Ask for a reference to the value held by the carrier.
/// let ref_one = carrier.create_ref().unwrap();
/// assert_eq!(*ref_one.lock().unwrap(), 7);
///
/// // Reference returned by Carrier can be sent to another thread.
/// std::thread::spawn(move || *ref_one.lock().unwrap() = 8usize);
///
/// // Close the carrier, no new references can be created.
/// carrier.close();
/// assert!(carrier.create_ref().is_none());
///
/// // Shutdown the carrier and wait for all references to be dropped.
/// // The value held by carrier is returned.
/// let mutex_value = carrier.wait();
/// // Destroy the mutex.
/// assert!(matches!(mutex_value.into_inner(), Ok(8usize)));
///
pub struct Carrier<T> {
    // Visible to tests.
    pub(self) template: Arc<CarrierTarget<T>>,
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

    pub fn create_ref(&self) -> Option<CarrierRef<T>> {
        if !self.shutdown.load(Ordering::Acquire) {
            Some(CarrierRef::new(&self.template))
        } else {
            None
        }
    }

    pub fn ref_count(&self) -> usize {
        *self.template.lock_count()
    }

    pub fn close(&self) {
        self.shutdown.store(true, Ordering::Release);
    }

    pub fn is_closed(&self) -> bool {
        self.shutdown.load(Ordering::Acquire)
    }

    fn unwrap_or_panic(self) -> T {
        let arc = self.template;
        assert_eq!(
            Arc::strong_count(&arc),
            1,
            "The carrier should not more than one outstanding Arc"
        );

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

/// A reference to an object owned by a [`Carrier`](`Carrier`).
///
/// The target will be alive for as long as this reference is alive.
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

#[cfg(test)]
mod tests {
    use crate::Carrier;
    use std::cell::RefCell;
    use std::time::Duration;

    #[test]
    fn test_basics() {
        let carrier = Carrier::new(7usize);

        let ref_one = carrier.create_ref().unwrap();
        let ref_two = carrier.create_ref().unwrap();
        // Carrier should be send.
        let (ref_three, carrier) =
            std::thread::spawn(|| (carrier.create_ref(), carrier))
                .join()
                .expect("Thread creation should never fail");
        let ref_three = ref_three.unwrap();

        assert_eq!(*ref_one, 7usize);
        assert_eq!(*ref_two, 7usize);
        assert_eq!(*ref_three, 7usize);

        carrier.close();
        assert!(carrier.is_closed());
        // Double close is OK.
        carrier.close();
        assert!(carrier.is_closed());

        assert!(carrier.create_ref().is_none());
        // Create should always fail.
        assert!(carrier.create_ref().is_none());

        assert_eq!(carrier.ref_count(), 3);

        let carrier =
            carrier.wait_timeout(Duration::from_micros(1)).expect_err(
                "Wait should not be successful \
                since there are outstanding references",
            );

        drop(ref_one);
        assert_eq!(carrier.ref_count(), 2);
        drop(ref_two);
        assert_eq!(carrier.ref_count(), 1);
        drop(ref_three);
        assert_eq!(carrier.ref_count(), 0);
        assert_eq!(carrier.wait(), 7usize);
    }

    #[test]
    #[should_panic]
    fn test_panic_outstanding_arc() {
        let carrier = Carrier::new(7usize);
        let _outstanding_ref = carrier.template.clone();

        // Carrier should panic when it sees an outstanding Arc.
        carrier.wait();
    }

    #[test]
    fn test_ref() {
        let carrier = Carrier::new(RefCell::new(7usize));
        let ref_one = carrier.create_ref().unwrap();
        let ref_two = carrier.create_ref().unwrap();

        *ref_two.borrow_mut() += 1;
        assert_eq!(8, *ref_one.borrow());

        *ref_one.borrow_mut() += 1;
        assert_eq!(9, *ref_two.borrow());
    }
}
