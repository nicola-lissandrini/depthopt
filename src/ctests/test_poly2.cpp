#include <iostream>
#include <cxxabi.h>
/// @brief Shortcut to cout line
/// @ingroup qdt
#define COUT(str) {std::cout <<  (str) << std::endl;}
/// @brief Print name and value of expression
/// @ingroup qdt
#define COUTN(var) {std::cout << "\e[33m" << #var << "\e[0m" << std::endl << var << std::endl;}
/// @brief Print name and value of an expression that is a time point
/// @ingroup qdt
#define COUTNT(var) {std::cout << "\e[33m" << #var << "\e[0m" << std::endl << printTime(var) << std::endl;}
/// @brief Print name and shape of expressione
/// @ingroup qdt
#define COUTNS(var) {COUTN(var.sizes());}
/// @brief Print calling function, name and value of expression
/// @ingroup qdt
#define COUTNF(var) {std::cout << "\e[32m" << __PRETTY_FUNCTION__ << "\n\e[33m" << #var << "\e[0m" << std::endl << var << std::endl;}
/// @brief Shortcut for printing name and value and returning the result of expression
/// @ingroup qdt
#define COUT_RET(var) {auto __ret = (var); COUTN(var); return __ret;}
/// @brief Get demangled type of expression
/// @ingroup qdt
#define TYPE(type) (abi::__cxa_demangle(typeid(type).name(), NULL,NULL,NULL))
/// @brief Shortcut for printing boost stacktrace
/// @ingroup qdt
#define STACKTRACE {std::cout << boost::stacktrace::stacktrace() << std::endl;}
/// @brief Print calling function and current file and line
/// @ingroup qdt
#define QUA {std::cout << "\e[33mReached " << __PRETTY_FUNCTION__ << "\e[0m:" << __LINE__ << std::endl; }


using namespace std;

class A
{
	int a;

public:
	A():
		 a(3)
	{}

	virtual void foo () {
		cout << a << endl;
	}
};

class B : public A
{
	int b;

public:
	B():
		 b(4)
	{}

private:
	void foo () final {
		cout << b << endl;
	}
};

template<class Derived>
class AT
{
	int a;

public:
	AT():
		 a(33)
	{}

	void foo () const {
		cout << "super" << endl;

		COUTN(TYPE(decltype(&AT<Derived>::derived)));
		derived().foo ();
	}

	const Derived &derived () const {
		return *static_cast<const Derived*> (this);
	}
};

class BT : public AT<BT>
{
	using Base = AT<BT>;
	friend Base;

	int b;
	std::string ciao;

public:
	explicit BT(int what = 44):
		 b(what),
		 ciao(std::to_string (b+100))
	{}

	BT(const BT &) = default;

private:
	void foo  () const {
		cout << b << endl;
		cout << ciao << endl;
	}
};



template<typename Derived>
void dump(const AT<Derived> &aa) {
	aa.foo ();
}

int main ()
{
	BT bb;
	BT bb2(bb);
	BT bb3 = bb;

	dump (bb);
	dump (bb2);
	dump (bb3);
}
