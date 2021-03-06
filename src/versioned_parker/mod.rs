use std::ops::{Deref, DerefMut};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, WaitTimeoutResult};
use std::time::Duration;

/// A thread parking and locking primitive that provide version numbers.
///
/// Like an [`std::sync::Condvar`], `VersionedParker` provides a `wait`
/// method and several `notify` methods. The `wait` method blocks the current
/// thread, while the `notify` methods unblocks waiting threads. Each time
/// `notify` is called, the parker version is increased. When a blocked thread
/// wakes up, it can check the internal counter and learn how many times it has
/// been notified. The version can be obtained by calling method
/// [`VersionedParker::version()`].
///
/// `VersionedParker` holds a piece of data that can be modified during `notify`
/// and `wait` operations. The data is versioned also versioned by the same
/// parker version.
///
/// ```
/// use more_sync::VersionedParker;
///
/// let versioned_parker = VersionedParker::new(0);
/// let mut guard = versioned_parker.lock();
///
/// let parker_clone = versioned_parker.clone();
/// std::thread::spawn(move || {
///     parker_clone.notify_one_mutate(|i| *i = 16);
///     assert_eq!(parker_clone.version(), 1);
///     // Version is 1, try_notify_all() should fail.
///     assert!(!parker_clone.try_notify_all(0));
/// });
///
/// guard.wait();
/// assert_eq!(guard.notified_count(), 1);
/// assert_eq!(*guard, 16);
/// ```
#[derive(Default, Clone, Debug)]
pub struct VersionedParker<T> {
    inner: Arc<Inner<T>>,
}

#[derive(Default, Debug)]
struct Inner<T> {
    version: AtomicUsize,
    data: Mutex<T>,
    condvar: Condvar,
}

impl<T> Inner<T> {
    fn version(&self) -> usize {
        self.version.load(Ordering::Acquire)
    }
}

impl<T> VersionedParker<T> {
    /// Creates a new `VersionedParker`, with the initial version being `0`, and
    /// the shared data being `data`.
    pub fn new(data: T) -> Self {
        Self {
            inner: Arc::new(Inner {
                version: AtomicUsize::new(0),
                data: Mutex::new(data),
                condvar: Condvar::new(),
            }),
        }
    }

    /// Locks the shared data and the version.
    ///
    /// A thread can then call [`VersionedGuard::wait()`] to wait for version
    /// changes.
    pub fn lock(&self) -> VersionedGuard<T> {
        let guard = self.inner.data.lock().unwrap();
        VersionedGuard {
            parker: self.inner.as_ref(),
            guard: Some(guard),
            notified_count: 0,
        }
    }

    fn do_notify(
        &self,
        expected_version: Option<usize>,
        mutate: fn(&mut T),
        notify: fn(&Condvar),
    ) -> bool {
        let mut guard = self.inner.data.lock().unwrap();
        if expected_version
            .map(|v| v == self.version())
            .unwrap_or(true)
        {
            self.inner.version.fetch_add(1, Ordering::AcqRel);
            mutate(guard.deref_mut());
            notify(&self.inner.condvar);
            return true;
        }
        false
    }

    /// Increases the version and notifies one blocked thread.
    pub fn notify_one(&self) {
        self.do_notify(None, |_| {}, Condvar::notify_one);
    }

    /// Increases the version, mutates the shared data and notifies one blocked
    /// thread.
    pub fn notify_one_mutate(&self, mutate: fn(&mut T)) {
        self.do_notify(None, mutate, Condvar::notify_one);
    }

    /// Increases the version and notifies one blocked thread, if the current
    /// version is `expected_version`.
    ///
    /// Returns `true` if the version matches.
    pub fn try_notify_one(&self, expected_version: usize) -> bool {
        self.do_notify(Some(expected_version), |_| {}, Condvar::notify_one)
    }

    /// Increases the version, modifies the shared data and notifies one blocked
    /// thread, if the current version is `expected_version`.
    ///
    /// Returns `true` if the version matches.
    pub fn try_notify_one_mutate(
        &self,
        expected_version: usize,
        mutate: fn(&mut T),
    ) -> bool {
        self.do_notify(Some(expected_version), mutate, Condvar::notify_one)
    }

    /// Increases the version and notifies all blocked threads.
    pub fn notify_all(&self) {
        self.do_notify(None, |_| {}, Condvar::notify_all);
    }

    /// Increases the version, modifies the shared data and notifies all blocked
    /// threads.
    pub fn notify_all_mutate(&self, mutate: fn(&mut T)) {
        self.do_notify(None, mutate, Condvar::notify_all);
    }

    /// Increases the version and notifies all blocked threads, if the current
    /// version is `expected_version`.
    ///
    /// Returns `true` if the version matches.
    pub fn try_notify_all(&self, expected_version: usize) -> bool {
        self.do_notify(Some(expected_version), |_| {}, Condvar::notify_all)
    }

    /// Increases the version, modifies the shared data and notifies all blocked
    /// threads, if the current version is `expected_version`.
    ///
    /// Returns `true` if the version matches.
    pub fn try_notify_all_mutate(
        &self,
        expected_version: usize,
        mutate: fn(&mut T),
    ) -> bool {
        self.do_notify(Some(expected_version), mutate, Condvar::notify_all)
    }

    /// Returns the current version.
    pub fn version(&self) -> usize {
        self.inner.version()
    }
}

/// Mutex guard returned by [`VersionedParker::lock`].
#[derive(Debug)]
pub struct VersionedGuard<'a, T> {
    parker: &'a Inner<T>,
    guard: Option<MutexGuard<'a, T>>,
    notified_count: usize,
}

impl<'a, T> VersionedGuard<'a, T> {
    /// Returns the current version.
    ///
    /// The version will not change unless [`wait()`](`VersionedGuard::wait`) or
    /// [`wait_timeout()`](`VersionedGuard::wait_timeout`) is called.
    pub fn version(&self) -> usize {
        self.parker.version()
    }

    /// Returns if we were notified during last period.
    ///
    /// If we never waited, `notified()` returns false.
    pub fn notified(&self) -> bool {
        self.notified_count != 0
    }

    /// Returns the number of times we were notified during last wait.
    ///
    /// If we never waited, `notification_count()` returns 0.
    pub fn notified_count(&self) -> usize {
        self.notified_count
    }

    /// Blocks the current thread until notified.
    ///
    /// `wait()` updates the version stored in this guard.
    pub fn wait(&mut self) {
        let guard = self.guard.take().unwrap();
        let version = self.parker.version();

        self.guard = Some(self.parker.condvar.wait(guard).unwrap());
        self.notified_count = self.parker.version() - version;
    }

    /// Blocks the current thread until notified, for up to `timeout`.
    ///
    /// `wait_timeout()` updates the version stored in this guard.
    pub fn wait_timeout(&mut self, timeout: Duration) -> WaitTimeoutResult {
        let guard = self.guard.take().unwrap();
        let version = self.parker.version();
        let (guard_result, wait_result) =
            self.parker.condvar.wait_timeout(guard, timeout).unwrap();

        self.guard = Some(guard_result);
        self.notified_count = self.parker.version() - version;

        wait_result
    }
}

impl<'a, T> Deref for VersionedGuard<'a, T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        self.guard.as_deref().unwrap()
    }
}

impl<'a, T> DerefMut for VersionedGuard<'a, T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.guard.as_deref_mut().unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basics() {
        let versioned_parker = VersionedParker::new(0);
        assert_eq!(versioned_parker.version(), 0);

        versioned_parker.notify_one();
        assert_eq!(versioned_parker.version(), 1);

        versioned_parker.notify_one_mutate(|i| *i = 32);
        let mut guard = versioned_parker.lock();
        assert_eq!(versioned_parker.version(), 2);
        assert_eq!(*guard, 32);

        let parker_clone = versioned_parker.clone();
        std::thread::spawn(move || parker_clone.notify_one_mutate(|i| *i = 64));
        guard.wait();

        assert_eq!(guard.notified_count(), 1);
        assert_eq!(*guard, 64);
    }

    #[test]
    fn test_multiple_notify() {
        let versioned_parker = VersionedParker::new(0);
        let mut guard = versioned_parker.lock();

        let parker_clone = versioned_parker.clone();
        std::thread::spawn(move || {
            parker_clone.notify_all();
            parker_clone.notify_all_mutate(|i| *i = 128);
            parker_clone.notify_one_mutate(|i| *i = 256);
            parker_clone.notify_one_mutate(|i| *i = 512);
        });

        guard.wait();
        let expected_value = match guard.notified_count() {
            1 => 0,
            2 => 128,
            3 => 256,
            4 => 512,
            _ => panic!("notify count should not be larger than 3"),
        };
        assert_eq!(*guard, expected_value);
    }

    #[test]
    fn test_try_notify() {
        let versioned_parker = VersionedParker::new(0);
        let mut guard = versioned_parker.lock();

        let parker_clone = versioned_parker.clone();
        std::thread::spawn(move || {
            assert!(parker_clone.try_notify_one(0));
            assert!(!parker_clone.try_notify_all(0));
        });

        guard.wait();
        assert_eq!(guard.notified_count(), 1);
        assert_eq!(*guard, 0);
    }

    #[test]
    fn test_try_notify_mutate() {
        let versioned_parker = VersionedParker::new(0);
        let mut guard = versioned_parker.lock();

        let parker_clone = versioned_parker.clone();
        std::thread::spawn(move || {
            assert!(parker_clone.try_notify_one_mutate(0, |i| *i = 1024));
            assert!(!parker_clone.try_notify_all_mutate(0, |i| *i = 2048));
        });

        guard.wait();
        assert_eq!(guard.notified_count(), 1);
        assert_eq!(*guard, 1024);
    }
}
