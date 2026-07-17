import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import Navbar from '../components/Navbar.jsx';
import toast from 'react-hot-toast';
import api from '../services/api';

function PatientSearch() {
  const [searchTerm, setSearchTerm] = useState('');
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false);
  const navigate = useNavigate();

  const handleSearch = async (e) => {
    e.preventDefault();
    if (!searchTerm.trim()) {
      toast.error('Please enter a search term');
      return;
    }
    setLoading(true);
    try {
      const mockResults = [
        { id: 'P001', name: 'John Doe', room: 'ICU-101', risk: 'HIGH' },
        { id: 'P002', name: 'Jane Smith', room: 'ICU-102', risk: 'MEDIUM' },
        { id: 'P003', name: 'Robert Johnson', room: 'ICU-103', risk: 'LOW' },
      ];
      setResults(mockResults.filter(p => 
        p.id.includes(searchTerm) || p.name.toLowerCase().includes(searchTerm.toLowerCase())
      ));
      if (results.length === 0) {
        toast.success('No patients found');
      }
    } catch (error) {
      toast.error('Search failed');
    } finally {
      setLoading(false);
    }
  };

  const getRiskColor = (risk) => {
    switch(risk) {
      case 'HIGH': return 'bg-red-100 text-red-800';
      case 'MEDIUM': return 'bg-yellow-100 text-yellow-800';
      default: return 'bg-green-100 text-green-800';
    }
  };

  return (
    <div className="min-h-screen bg-gray-50">
      <Navbar />
      <div className="container mx-auto px-4 py-8">
        <h1 className="text-2xl font-bold mb-6">Patient Search</h1>
        
        <form onSubmit={handleSearch} className="flex gap-4 mb-8">
          <input
            type="text"
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            placeholder="Search by Patient ID or Name"
            className="flex-1 input-field"
          />
          <button type="submit" className="btn-primary px-8">
            {loading ? 'Searching...' : '🔍 Search'}
          </button>
        </form>

        {results.length > 0 && (
          <div className="bg-white rounded-xl shadow-lg overflow-hidden">
            <table className="w-full">
              <thead className="bg-gray-50">
                <tr>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Patient ID</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Name</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Room</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Risk Status</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Action</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-200">
                {results.map((patient) => (
                  <tr key={patient.id} className="hover:bg-gray-50">
                    <td className="px-6 py-4 text-sm font-medium text-gray-900">{patient.id}</td>
                    <td className="px-6 py-4 text-sm text-gray-600">{patient.name}</td>
                    <td className="px-6 py-4 text-sm text-gray-600">{patient.room}</td>
                    <td className="px-6 py-4">
                      <span className={`px-2 py-1 rounded-full text-xs font-semibold ${getRiskColor(patient.risk)}`}>
                        {patient.risk}
                      </span>
                    </td>
                    <td className="px-6 py-4">
                      <button
                        onClick={() => navigate(`/patients/${patient.id}`)}
                        className="text-primary-600 hover:text-primary-800 text-sm font-medium"
                      >
                        View Details
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}

        {results.length === 0 && searchTerm && !loading && (
          <div className="text-center text-gray-500 py-12">
            <p className="text-lg">No patients found</p>
          </div>
        )}
      </div>
    </div>
  );
}

export default PatientSearch;
