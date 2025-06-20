#ifndef BODY_H
#define BODY_H
#include <Vector.h> /// This is in GameDev
using namespace MATH; 

/// Just forward declair these classes so I can define a pointer to them
/// Used later in the course
class Mesh;
class Texture;

class Body {
public:
    Body();
    ~Body();
private: /// Physics stuff
	Vec3 pos;
	Vec3 vel;
	Vec3 accel;
	float mass;
	float radius = 1.0f; // Set to your desired ball size
private: /// Graphics stuff 
	Mesh *mesh;
	Texture *texture;
	
public:
	
	Vec3 GetPosition() const { return pos; }
	float GetRadius() const { return radius; }
	bool OnCreate();
	void OnDestroy();
	void Update(float deltaTime);
	void Render() const;
	void SetPosition(const Vec3& position);
	void SetRadius(float r) { radius = r; }
	void ApplyForce(Vec3 force);
	void setAccel(const Vec3 &accel_) { accel = accel_;}

	
};

#endif
