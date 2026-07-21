import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import toast from 'react-hot-toast';
import AdminSidebar from '../../components/AdminSidebar.jsx';
import Breadcrumb from '../../components/Breadcrumb.jsx';
import api from '../../services/api';
import { 
  Search, UserPlus, Trash2, Shield, ShieldOff, 
  UserCheck, UserX, RefreshCw, AlertTriangle,
  Edit2, Check, X
} from 'lucide-react';

function Users() {
  const [users, setUsers] = useState([]);
  const [loading, setLoading] = useState(true);
  const [searchTerm, setSearchTerm] = useState('');
  const [showAddModal, setShowAddModal] = useState(false);
  const [showDeleteModal, setShowDeleteModal] = useState(false);
  const [userToDelete, setUserToDelete] = useState(null);
  const [editingUser, setEditingUser] = useState(null);
  const [formData, setFormData] = useState({
    username: '',
    email: '',
    full_name: '',
    password: '',
    role: 'clinician',
    is_admin: false
  });
  const navigate = useNavigate();

  useEffect(() => {
    loadUsers();
  }, []);

  const loadUsers = async () => {
    setLoading(true);
    try {
      const response = await api.getUsers();
      console.log('Users response:', response.data);
      const userData = response.data?.users || response.data || [];
      setUsers(Array.isArray(userData) ? userData : []);
    } catch (error) {
      console.error('Failed to load users:', error);
      toast.error('Failed to load users');
      if (error.response?.status === 403) {
        navigate('/dashboard');
      }
      // Set demo data if API fails
      setUsers([
        { id: '1', username: 'admin', email: 'admin@icupredictor.com', full_name: 'System Administrator', is_admin: true, is_active: true, created_at: new Date().toISOString() },
        { id: '2', username: 'doctor', email: 'doctor@hospital.com', full_name: 'Dr. Test User', is_admin: false, is_active: true, created_at: new Date().toISOString() }
      ]);
    } finally {
      setLoading(false);
    }
  };

  const handleToggleAdmin = async (user) => {
    try {
      const newAdminStatus = !user.is_admin;
      await api.makeAdmin(user.id);
      
      setUsers(prev => prev.map(u => 
        u.id === user.id ? { ...u, is_admin: newAdminStatus } : u
      ));
      
      toast.success(`${user.username} is now ${newAdminStatus ? 'an Admin' : 'a Clinician'}`);
    } catch (error) {
      console.error('Failed to update role:', error);
      toast.error('Failed to update user role');
    }
  };

  const handleToggleStatus = async (user) => {
    try {
      const newStatus = !user.is_active;
      await api.updateUser(user.id, { is_active: newStatus });
      
      setUsers(prev => prev.map(u => 
        u.id === user.id ? { ...u, is_active: newStatus } : u
      ));
      
      toast.success(`${user.username} ${newStatus ? 'activated' : 'deactivated'}`);
    } catch (error) {
      console.error('Failed to update status:', error);
      toast.error('Failed to update user status');
    }
  };

  const handleDeleteClick = (user) => {
    setUserToDelete(user);
    setShowDeleteModal(true);
  };

  const handleConfirmDelete = async () => {
    if (!userToDelete) return;
    
    try {
      await api.deleteUser(userToDelete.id);
      setUsers(prev => prev.filter(u => u.id !== userToDelete.id));
      toast.success('User deleted successfully!');
      setShowDeleteModal(false);
      setUserToDelete(null);
    } catch (error) {
      console.error('Failed to delete user:', error);
      toast.error('Failed to delete user');
    }
  };

  const handleCancelDelete = () => {
    setShowDeleteModal(false);
    setUserToDelete(null);
  };

  const handleAddUser = async (e) => {
    e.preventDefault();
    try {
      if (formData.password && formData.password.length < 6) {
        toast.error('Password must be at least 6 characters');
        return;
      }

      const newUser = {
        username: formData.username,
        email: formData.email,
        full_name: formData.full_name,
        password: formData.password || 'Temp123!',
        is_admin: formData.role === 'admin'
      };

      await api.register(newUser);
      toast.success('User added successfully!');
      setShowAddModal(false);
      resetForm();
      loadUsers();
    } catch (error) {
      console.error('Failed to add user:', error);
      toast.error('Failed to add user');
    }
  };

  const resetForm = () => {
    setFormData({
      username: '',
      email: '',
      full_name: '',
      password: '',
      role: 'clinician',
      is_admin: false
    });
    setEditingUser(null);
  };

  const filteredUsers = users.filter(user =>
    user.username?.toLowerCase().includes(searchTerm.toLowerCase()) ||
    user.email?.toLowerCase().includes(searchTerm.toLowerCase()) ||
    user.full_name?.toLowerCase().includes(searchTerm.toLowerCase())
  );

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600"></div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-100">
      <AdminSidebar />
      <div className="ml-64 p-8">
        <Breadcrumb />
        <div className="flex justify-between items-center mb-6">
          <div>
            <h1 className="text-2xl font-bold">User Management</h1>
            <p className="text-sm text-gray-500">Manage system users and permissions</p>
          </div>
          <div className="flex gap-3">
            <button
              onClick={loadUsers}
              className="flex items-center gap-2 px-4 py-2 bg-gray-600 text-white rounded-lg hover:bg-gray-700 transition"
            >
              <RefreshCw className="w-4 h-4" />
              Refresh
            </button>
            <button
              onClick={() => { resetForm(); setShowAddModal(true); }}
              className="flex items-center gap-2 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition"
            >
              <UserPlus className="w-4 h-4" />
              Add User
            </button>
          </div>
        </div>

        <div className="mb-6">
          <div className="relative">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400" />
            <input
              type="text"
              placeholder="Search users by name, username or email..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="w-full md:w-96 pl-10 pr-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 outline-none"
            />
          </div>
        </div>

        <div className="bg-white rounded-xl shadow overflow-hidden">
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead className="bg-gray-50">
                <tr>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">User</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Username</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Email</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Role</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Status</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Actions</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-200">
                {filteredUsers.length === 0 ? (
                  <tr>
                    <td colSpan="6" className="px-6 py-8 text-center text-gray-500">
                      No users found
                    </td>
                  </tr>
                ) : (
                  filteredUsers.map((user) => (
                    <tr key={user.id} className="hover:bg-gray-50 transition-colors">
                      <td className="px-6 py-4 font-medium text-gray-900">{user.full_name || user.username}</td>
                      <td className="px-6 py-4 text-gray-600">{user.username}</td>
                      <td className="px-6 py-4 text-gray-600">{user.email}</td>
                      <td className="px-6 py-4">
                        <span className={`px-2 py-1 rounded-full text-xs font-semibold ${
                          user.is_admin ? 'bg-purple-100 text-purple-800' : 'bg-gray-100 text-gray-800'
                        }`}>
                          {user.is_admin ? 'Admin' : 'Clinician'}
                        </span>
                      </td>
                      <td className="px-6 py-4">
                        <span className={`px-2 py-1 rounded-full text-xs font-semibold ${
                          user.is_active ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'
                        }`}>
                          {user.is_active ? 'Active' : 'Inactive'}
                        </span>
                      </td>
                      <td className="px-6 py-4">
                        <div className="flex items-center gap-2">
                          <button
                            onClick={() => handleToggleAdmin(user)}
                            className={`p-1.5 rounded-lg transition-colors ${
                              user.is_admin 
                                ? 'text-purple-600 hover:bg-purple-50' 
                                : 'text-gray-400 hover:bg-gray-100'
                            }`}
                            title={user.is_admin ? 'Remove Admin' : 'Make Admin'}
                          >
                            {user.is_admin ? <ShieldOff className="w-4 h-4" /> : <Shield className="w-4 h-4" />}
                          </button>
                          <button
                            onClick={() => handleToggleStatus(user)}
                            className={`p-1.5 rounded-lg transition-colors ${
                              user.is_active 
                                ? 'text-green-600 hover:bg-green-50' 
                                : 'text-red-600 hover:bg-red-50'
                            }`}
                            title={user.is_active ? 'Deactivate' : 'Activate'}
                          >
                            {user.is_active ? <UserX className="w-4 h-4" /> : <UserCheck className="w-4 h-4" />}
                          </button>
                          <button
                            onClick={() => handleDeleteClick(user)}
                            className="p-1.5 text-red-600 hover:bg-red-50 rounded-lg transition-colors"
                            title="Delete User"
                          >
                            <Trash2 className="w-4 h-4" />
                          </button>
                        </div>
                      </td>
                    </tr>
                  ))
                )}
              </tbody>
            </table>
          </div>
        </div>

        <div className="mt-4 text-sm text-gray-500">
          Total Users: {filteredUsers.length}
        </div>
      </div>

      {/* Delete Confirmation Modal */}
      {showDeleteModal && userToDelete && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white rounded-2xl p-6 w-full max-w-md mx-4">
            <div className="flex items-center gap-3 mb-4">
              <div className="p-3 bg-red-100 rounded-full">
                <Trash2 className="w-6 h-6 text-red-600" />
              </div>
              <div>
                <h2 className="text-xl font-bold text-gray-900">Delete User</h2>
                <p className="text-sm text-gray-500">This action cannot be undone</p>
              </div>
            </div>
            
            <div className="bg-red-50 border border-red-200 rounded-lg p-4 mb-6">
              <div className="flex items-start gap-3">
                <AlertTriangle className="w-5 h-5 text-red-600 flex-shrink-0 mt-0.5" />
                <div>
                  <p className="text-sm text-red-800 font-medium">Are you sure you want to delete this user?</p>
                  <p className="text-sm text-red-600 mt-1">
                    <strong>User:</strong> {userToDelete.full_name || userToDelete.username}
                    <br />
                    <strong>Username:</strong> {userToDelete.username}
                    <br />
                    <strong>Email:</strong> {userToDelete.email}
                  </p>
                  <p className="text-xs text-red-500 mt-2">
                    All associated data and permissions will be permanently removed.
                  </p>
                </div>
              </div>
            </div>
            
            <div className="flex justify-end gap-3">
              <button
                onClick={handleCancelDelete}
                className="px-4 py-2 border border-gray-300 rounded-lg hover:bg-gray-50 transition"
              >
                Cancel
              </button>
              <button
                onClick={handleConfirmDelete}
                className="px-4 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700 transition flex items-center gap-2"
              >
                <Trash2 className="w-4 h-4" />
                Delete User
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Add User Modal */}
      {showAddModal && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
          <div className="bg-white rounded-2xl p-6 w-full max-w-md max-h-[90vh] overflow-y-auto">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-xl font-bold">Add New User</h2>
              <button
                onClick={() => { setShowAddModal(false); resetForm(); }}
                className="text-gray-400 hover:text-gray-600"
              >
                <X className="w-5 h-5" />
              </button>
            </div>
            <form onSubmit={handleAddUser}>
              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">Full Name *</label>
                  <input
                    type="text"
                    value={formData.full_name}
                    onChange={(e) => setFormData({...formData, full_name: e.target.value})}
                    className="w-full px-4 py-2 border border-gray-300 rounded-xl focus:ring-2 focus:ring-blue-500 outline-none"
                    required
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">Username *</label>
                  <input
                    type="text"
                    value={formData.username}
                    onChange={(e) => setFormData({...formData, username: e.target.value})}
                    className="w-full px-4 py-2 border border-gray-300 rounded-xl focus:ring-2 focus:ring-blue-500 outline-none"
                    required
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">Email *</label>
                  <input
                    type="email"
                    value={formData.email}
                    onChange={(e) => setFormData({...formData, email: e.target.value})}
                    className="w-full px-4 py-2 border border-gray-300 rounded-xl focus:ring-2 focus:ring-blue-500 outline-none"
                    required
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">Password</label>
                  <input
                    type="password"
                    value={formData.password}
                    onChange={(e) => setFormData({...formData, password: e.target.value})}
                    className="w-full px-4 py-2 border border-gray-300 rounded-xl focus:ring-2 focus:ring-blue-500 outline-none"
                    placeholder="Leave blank for default: Temp123!"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">Role</label>
                  <select
                    value={formData.role}
                    onChange={(e) => setFormData({...formData, role: e.target.value})}
                    className="w-full px-4 py-2 border border-gray-300 rounded-xl focus:ring-2 focus:ring-blue-500 outline-none"
                  >
                    <option value="clinician">Clinician</option>
                    <option value="admin">Admin</option>
                  </select>
                </div>
              </div>
              <div className="flex justify-end gap-3 mt-6">
                <button
                  type="button"
                  onClick={() => { setShowAddModal(false); resetForm(); }}
                  className="px-4 py-2 border border-gray-300 rounded-xl hover:bg-gray-50 transition"
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  className="px-6 py-2 bg-blue-600 text-white rounded-xl hover:bg-blue-700 transition"
                >
                  Add User
                </button>
              </div>
            </form>
          </div>
        </div>
      )}
    </div>
  );
}

export default Users;
