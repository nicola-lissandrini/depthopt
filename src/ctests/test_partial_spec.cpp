#include <iostream>
#include <boost/optional.hpp>
#include <type_traits>

using namespace std;

template<typename T>
struct empty_container {
	T value;

	empty_container (T &&_value):
		 value(_value)
	{}

	operator T () {
		return value;
	}
	T &operator = (const T &_value) {
		value = value;
		return value;
	}
};

class A {
	template<typename T, template <typename...> typename container = empty_container>
	struct Type {
		container<T> convert ();
	};

public:
	template<typename T>
	T get () {
		return get<T, empty_container> ();
	}

	template<typename T, template <typename...> typename container>
	container<T> get ()
	{
		Type<T, container> type;

		return type.convert ();
	}
};

template<typename T>
struct A::Type<T, vector> {
	vector<T> convert () {
		return {1,2};
	}
};

template<>
struct A::Type<int> {
	empty_container<int> convert () {
		return 3;
	}
};

int main () {
	A ciao;

	int a = ciao.get<int> ();
	vector<int> b = ciao.get<int, vector> ();

	cout << a << endl;
	cout << b[0] << " " << b[1] << endl;
}
